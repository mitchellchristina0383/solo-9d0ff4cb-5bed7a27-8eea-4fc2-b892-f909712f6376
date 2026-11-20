"""Backup expansion what-if planning — computes a plan, never copies anything.

Training checkpoints are stored as per-rank shards, weights and momentum on
the same volume.  To survive a whole rack outage, every shard needs replicas
on at least two racks.  Given a batch of *completed* reshard jobs and a set
of backup volumes (rack, remaining bytes, complete shards already held), this
module works out which single-rack shards need one extra replica, picks the
target volume for each new replica, and reports the minimum number of bytes
the one expandable volume must grow.  Every other volume stays within its
remaining bytes.

Shard size comes straight from the job's rank interval: ``(end - start)``
float32 elements for weights **and** momentum, i.e. ``(end - start) * 4 * 2``
bytes.

The placement search is exact (dynamic programming), so the input is bounded
to keep it instant: at most ``MAX_BACKFILL_SHARDS`` shards to backfill and
``MAX_VOLUMES`` volumes per run.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .jobs import Job
from .manifest import FLOAT32_BYTES

SHARD_PARTS = 2  # weights + momentum are stored on the same volume
MAX_BACKFILL_SHARDS = 10
MAX_VOLUMES = 5

# Machine-readable reason codes for shards the plan cannot cover.
NO_SOURCE_REPLICA = "no_source_replica"
NO_ELIGIBLE_RACK = "no_eligible_rack"
INSUFFICIENT_CAPACITY = "insufficient_capacity"
CAPACITY_CONTENTION = "capacity_contention"

_INF = 1 << 62  # "infinite" expandable-bytes cost for the DP


class BackupPlanError(ValueError):
    """Invalid plan request (the API maps it to HTTP 422)."""


@dataclass(frozen=True)
class VolumeInput:
    """One backup volume as declared by the admin.

    ``free_bytes`` is the space remaining *right now* (already accounting for
    ``held``); ``held`` lists the complete shards stored on the volume and is
    only used to know which racks hold which shard.
    """

    volume_id: str
    rack: str
    free_bytes: int
    held: tuple[tuple[str, int], ...]  # (job_id, rank) of complete shards


@dataclass(frozen=True)
class _Volume:
    volume_id: str
    rack: str
    free_bytes: int
    expandable: bool


@dataclass(frozen=True)
class _BackfillShard:
    job_id: str
    rank: int
    size_bytes: int
    holders: tuple[int, ...]  # indices of volumes holding a replica, input order
    racks: frozenset[str]  # rack(s) of those holders — exactly one by construction


@dataclass(frozen=True)
class Placement:
    job_id: str
    rank: int
    size_bytes: int
    source_volume: str
    source_rack: str
    target_volume: str
    target_rack: str


@dataclass(frozen=True)
class Unplaced:
    job_id: str
    rank: int
    size_bytes: int
    code: str
    reason: str


@dataclass(frozen=True)
class VolumePlan:
    volume_id: str
    rack: str
    expandable: bool
    free_bytes: int
    planned_bytes: int  # new bytes this plan would add
    remaining_bytes: int  # free_bytes - planned_bytes (negative only if expandable)
    min_grow_bytes: int  # >0 only for the expandable volume


@dataclass(frozen=True)
class BackupPlan:
    feasible: bool  # true only if every shard that needs a replica got one
    job_ids: tuple[str, ...]
    total_shards: int
    redundant_shards: int  # already replicated on >= 2 racks
    backfill_shards: int  # replicas exist on exactly one rack
    placements: tuple[Placement, ...]
    unplaced: tuple[Unplaced, ...]
    volumes: tuple[VolumePlan, ...]
    expandable_volume: str
    min_expand_bytes: int  # minimum the expandable volume must grow


def shard_size_bytes(job: Job, rank: int) -> int:
    """Bytes one replica of a rank occupies: weights + momentum, same volume."""
    state = job.ranks[rank]
    return (state.end - state.start) * FLOAT32_BYTES * SHARD_PARTS


def _min_expandable_assignment(
    shards: list[_BackfillShard], volumes: list[_Volume]
) -> dict[int, int] | None:
    """Assign every shard to an eligible volume, minimising the bytes landed
    on the expandable volume.  Returns {shard position: volume index}, or
    None when no assignment places every shard.

    Exact DP over (shard index, remaining free bytes): with <= 10 shards and
    <= 5 volumes this is instant, and unlike a greedy pass it still finds the
    zero-expansion packing when shards contend for the same volume space.
    The expandable volume is modelled as unlimited; minimising the bytes on
    it also minimises its growth ``max(0, bytes - free)``.
    """
    eligible = [
        tuple(vi for vi, v in enumerate(volumes) if v.rack not in shard.racks)
        for shard in shards
    ]
    if any(not cand for cand in eligible):
        return None
    exp = next(vi for vi, v in enumerate(volumes) if v.expandable)
    n = len(shards)

    @lru_cache(maxsize=None)
    def dp(i: int, free: tuple[int, ...]) -> int:
        """Min bytes on the expandable volume to place shards[i:]."""
        if i == n:
            return 0
        size = shards[i].size_bytes
        best = _INF
        for vi in eligible[i]:
            if vi == exp:
                best = min(best, size + dp(i + 1, free))
            elif free[vi] >= size:
                nxt = free[:vi] + (free[vi] - size,) + free[vi + 1 :]
                best = min(best, dp(i + 1, nxt))
        return best

    free0 = tuple(v.free_bytes for v in volumes)
    if dp(0, free0) >= _INF:
        return None

    # Reconstruct one optimal assignment deterministically: among moves that
    # keep the optimum, take the tightest-fitting non-expandable volume first
    # and the expandable one last.
    assignment: dict[int, int] = {}
    free = free0
    for i, shard in enumerate(shards):
        size = shard.size_bytes
        target = dp(i, free)
        ordered = sorted(
            (vi for vi in eligible[i] if vi != exp and free[vi] >= size),
            key=lambda vi: (free[vi] - size, vi),
        )
        if exp in eligible[i]:
            ordered.append(exp)
        for vi in ordered:
            nxt = free if vi == exp else free[:vi] + (free[vi] - size,) + free[vi + 1 :]
            cost = size if vi == exp else 0
            if cost + dp(i + 1, nxt) == target:
                assignment[i] = vi
                free = nxt
                break
        else:  # pragma: no cover — dp() proved a move exists
            raise AssertionError("optimal reconstruction failed")
    return assignment


def plan_backup(
    jobs: dict[str, Job],
    job_ids: list[str],
    volumes: list[VolumeInput],
    expandable_volume: str,
) -> BackupPlan:
    """Build the backfill plan.  ``jobs`` must cover every id in ``job_ids``
    and each job must be complete.  Raises BackupPlanError on bad input."""
    if not job_ids:
        raise BackupPlanError("job_ids must not be empty")
    if not volumes:
        raise BackupPlanError("at least one volume is required")
    if len(volumes) > MAX_VOLUMES:
        raise BackupPlanError(
            f"too many volumes: {len(volumes)} exceeds the limit of {MAX_VOLUMES} per plan run"
        )
    ids = [v.volume_id for v in volumes]
    if len(set(ids)) != len(ids):
        dup = next(i for i in ids if ids.count(i) > 1)
        raise BackupPlanError(f"duplicate volume_id {dup!r}")
    if expandable_volume not in set(ids):
        raise BackupPlanError(
            f"expandable_volume {expandable_volume!r} is not one of the submitted volumes"
        )

    # Validate the declared shards and index which volumes hold each shard.
    holders: dict[tuple[str, int], list[int]] = {}  # (job_id, rank) -> volume indices
    for vi, vol in enumerate(volumes):
        seen: set[tuple[str, int]] = set()
        for jid, rank in vol.held:
            key = (jid, rank)
            if key in seen:
                raise BackupPlanError(
                    f"volume {vol.volume_id!r} lists shard ({jid}, rank {rank}) twice"
                )
            seen.add(key)
            if jid not in jobs:
                raise BackupPlanError(
                    f"volume {vol.volume_id!r} lists a shard of job {jid!r} which is "
                    "not part of this plan; include it in job_ids"
                )
            job = jobs[jid]
            if not 0 <= rank < job.world_size:
                raise BackupPlanError(
                    f"volume {vol.volume_id!r} lists rank {rank} of job {jid!r}, "
                    f"but the job only has ranks 0..{job.world_size - 1}"
                )
            holders.setdefault(key, []).append(vi)

    # Classify every shard of the selected jobs.
    total = 0
    redundant = 0
    unplaced: list[Unplaced] = []
    backfill: list[_BackfillShard] = []
    vols = [
        _Volume(v.volume_id, v.rack, v.free_bytes, v.volume_id == expandable_volume)
        for v in volumes
    ]
    for jid in job_ids:
        job = jobs[jid]
        for state in job.ranks:
            total += 1
            held_by = holders.get((jid, state.rank), [])
            racks = {vols[vi].rack for vi in held_by}
            size = shard_size_bytes(job, state.rank)
            if len(racks) >= 2:
                redundant += 1  # already survives a rack outage
            elif not held_by:
                unplaced.append(
                    Unplaced(
                        jid,
                        state.rank,
                        size,
                        NO_SOURCE_REPLICA,
                        "no source replica: no submitted volume holds this shard, "
                        "so there is nothing to copy from",
                    )
                )
            else:
                backfill.append(
                    _BackfillShard(jid, state.rank, size, tuple(held_by), frozenset(racks))
                )

    if len(backfill) > MAX_BACKFILL_SHARDS:
        raise BackupPlanError(
            f"{len(backfill)} shards need a cross-rack replica, above the limit of "
            f"{MAX_BACKFILL_SHARDS} per plan run; narrow the job selection"
        )

    # Split off shards that can never be placed, so they don't block the rest.
    placeable: list[_BackfillShard] = []
    for shard in backfill:
        cand = [v for v in vols if v.rack not in shard.racks]
        rack_list = ", ".join(sorted(shard.racks))
        if not cand:
            unplaced.append(
                Unplaced(
                    shard.job_id,
                    shard.rank,
                    shard.size_bytes,
                    NO_ELIGIBLE_RACK,
                    "no submitted volume sits on a different rack than the existing "
                    f"replica ({rack_list})",
                )
            )
        elif not any(v.expandable or v.free_bytes >= shard.size_bytes for v in cand):
            unplaced.append(
                Unplaced(
                    shard.job_id,
                    shard.rank,
                    shard.size_bytes,
                    INSUFFICIENT_CAPACITY,
                    f"no eligible volume can hold {shard.size_bytes} bytes: volumes on "
                    "other racks have less remaining and the expandable volume is not "
                    "on a different rack",
                )
            )
        else:
            placeable.append(shard)

    # Largest first: packs better and shrinks the search.
    ordered = sorted(placeable, key=lambda s: (-s.size_bytes, s.job_id, s.rank))
    assignment = _min_expandable_assignment(ordered, vols) if ordered else {}

    placements: list[Placement] = []
    planned = [0] * len(vols)
    if assignment is None:
        # Genuine contention: individually placeable, but not together.  Don't
        # pick arbitrary winners — report the whole group as unplaced.
        unplaced.extend(
            Unplaced(
                shard.job_id,
                shard.rank,
                shard.size_bytes,
                CAPACITY_CONTENTION,
                "volumes on other racks cannot hold all shards to backfill at once "
                "(capacity contention); free space, add a volume, or shrink the batch",
            )
            for shard in ordered
        )
    else:
        for pos, vi in assignment.items():
            shard = ordered[pos]
            src = vols[shard.holders[0]]
            placements.append(
                Placement(
                    job_id=shard.job_id,
                    rank=shard.rank,
                    size_bytes=shard.size_bytes,
                    source_volume=src.volume_id,
                    source_rack=src.rack,
                    target_volume=vols[vi].volume_id,
                    target_rack=vols[vi].rack,
                )
            )
            planned[vi] += shard.size_bytes

    volume_plans: list[VolumePlan] = []
    min_expand = 0
    for v, added in zip(vols, planned):
        grow = max(0, added - v.free_bytes) if v.expandable else 0
        if v.expandable:
            min_expand = grow
        volume_plans.append(
            VolumePlan(
                volume_id=v.volume_id,
                rack=v.rack,
                expandable=v.expandable,
                free_bytes=v.free_bytes,
                planned_bytes=added,
                remaining_bytes=v.free_bytes - added,
                min_grow_bytes=grow,
            )
        )

    placements.sort(key=lambda p: (p.job_id, p.rank))
    unplaced.sort(key=lambda u: (u.job_id, u.rank))
    return BackupPlan(
        feasible=not unplaced,
        job_ids=tuple(job_ids),
        total_shards=total,
        redundant_shards=redundant,
        backfill_shards=len(backfill),
        placements=tuple(placements),
        unplaced=tuple(unplaced),
        volumes=tuple(volume_plans),
        expandable_volume=expandable_volume,
        min_expand_bytes=min_expand,
    )
