"""Tests for POST /backup/plan — cross-rack backup expansion what-if.

Shard size recap: a replica is weights + momentum on the same volume, i.e.
``(end - start) * 4 * 2`` bytes.  With TOTAL=1000 and world=4 every rank owns
250 elements, so every shard below is 2000 bytes unless stated otherwise.
"""

import itertools
import random

import pytest
from fastapi.testclient import TestClient

from app.backup import _BackfillShard, _min_expandable_assignment, _Volume
from app.main import Settings, create_app
from tests.conftest import write_source

TOTAL = 1000
STEP = 321
WORLD = 4
SHARD_BYTES = 250 * 4 * 2  # 2000: weights + momentum on the same volume


@pytest.fixture()
def client(tmp_path):
    source = tmp_path / "src"
    write_source(
        source,
        total=TOTAL,
        step=STEP,
        weight_bounds=[0, 400, TOTAL],
        momentum_bounds=[0, 250, 800, TOTAL],
    )
    app = create_app(Settings(source_dir=source, output_root=tmp_path / "out",
                              chunk_bytes=256, sync_execution=True))
    with TestClient(app) as c:
        c.source = source
        yield c


def done_job(client, world=WORLD):
    resp = client.post("/reshard", json={"new_world_size": world})
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "done"  # sync execution in tests
    return body


def plan(client, job_ids, volumes, expandable):
    if isinstance(job_ids, str):
        job_ids = [job_ids]
    return client.post("/backup/plan", json={
        "job_ids": job_ids,
        "volumes": volumes,
        "expandable_volume": expandable,
    })


def vol(volume_id, rack, free_bytes, shards=()):
    return {"volume_id": volume_id, "rack": rack, "free_bytes": free_bytes,
            "shards": list(shards)}


def ref(job_id, rank):
    return {"job_id": job_id, "rank": rank}


def redundant_refs(job, keep=()):
    """Refs for every rank NOT in ``keep`` — tests put these on two racks so
    only the `keep` ranks drive the scenario (unlisted ranks would otherwise
    count as "no source replica")."""
    jid = job["job_id"]
    return [ref(jid, r["rank"]) for r in job["ranks"] if r["rank"] not in keep]


class TestOptimizerExactness:
    """The DP must match brute force when shards contend for volume space."""

    def test_matches_brute_force_on_random_contention(self):
        rng = random.Random(20261120)
        checked = 0
        for _ in range(150):
            racks = [f"rack-{i}" for i in range(rng.randint(2, 3))]
            n_vols = rng.randint(2, 4)
            exp_idx = rng.randrange(n_vols)
            vols = [
                _Volume(
                    volume_id=f"v{i}",
                    rack=rng.choice(racks),
                    free_bytes=rng.choice([0, 4, 8, 16, 32]),
                    expandable=(i == exp_idx),
                )
                for i in range(n_vols)
            ]
            shards = []
            for s in range(rng.randint(1, 6)):
                holders = tuple(i for i in range(n_vols) if rng.random() < 0.4)
                shard_racks = {vols[i].rack for i in holders}
                if len(shard_racks) != 1:
                    continue  # not a single-rack shard -> not the optimizer's job
                shards.append(_BackfillShard(
                    job_id="j", rank=s, size_bytes=rng.choice([4, 8, 12, 16]),
                    holders=holders, racks=frozenset(shard_racks)))
            if not shards:
                continue
            checked += 1
            got = _min_expandable_assignment(shards, vols)

            # Brute force over every assignment: min bytes on the expandable
            # volume, non-expandable volumes never over their remaining bytes.
            best = None
            cands = [[vi for vi, v in enumerate(vols) if v.rack not in sh.racks]
                     for sh in shards]
            if all(cands):
                for combo in itertools.product(*cands):
                    used = [0] * n_vols
                    for si, vi in enumerate(combo):
                        used[vi] += shards[si].size_bytes
                    if any(used[vi] > v.free_bytes
                           for vi, v in enumerate(vols) if not v.expandable):
                        continue
                    exp_bytes = sum(shards[si].size_bytes for si, vi in enumerate(combo)
                                    if vols[vi].expandable)
                    best = exp_bytes if best is None else min(best, exp_bytes)

            if best is None:
                assert got is None, "DP found a plan where brute force finds none"
            else:
                assert got is not None, "DP gave up though a feasible plan exists"
                exp_bytes = sum(shards[si].size_bytes for si, vi in got.items()
                                if vols[vi].expandable)
                assert exp_bytes == best  # optimal, not just feasible
                used = [0] * n_vols
                for si, vi in got.items():
                    assert vols[vi].rack not in shards[si].racks  # cross-rack
                    used[vi] += shards[si].size_bytes
                for vi, v in enumerate(vols):
                    if not v.expandable:
                        assert used[vi] <= v.free_bytes  # never over budget
        assert checked > 100  # make sure the random loop actually exercised


class TestBasicPlanning:
    def test_single_rack_shard_gets_cross_rack_replica(self, client):
        job = done_job(client)
        jid = job["job_id"]
        # All four shards sit on rack-a only; rack-b has room for all.
        resp = plan(client, jid, [
            vol("vol-a", "rack-a", 0, [ref(jid, r) for r in range(4)]),
            vol("vol-b", "rack-b", 4 * SHARD_BYTES),
        ], expandable="vol-b")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["feasible"] is True
        assert body["total_shards"] == 4
        assert body["backfill_shards"] == 4
        assert body["redundant_shards"] == 0
        assert len(body["placements"]) == 4
        for p in body["placements"]:
            assert p["target_volume"] == "vol-b"
            assert p["target_rack"] == "rack-b"
            assert p["source_volume"] == "vol-a"
            assert p["source_rack"] == "rack-a"
            # Size comes from the job interval and covers weights + momentum.
            assert p["size_bytes"] == SHARD_BYTES
        assert body["min_expand_bytes"] == 0
        vb = next(v for v in body["volumes"] if v["volume_id"] == "vol-b")
        assert vb["planned_bytes"] == 4 * SHARD_BYTES
        assert vb["remaining_bytes"] == 0

    def test_shard_size_follows_interval(self, client):
        # world=3 over 1000 elements -> uneven intervals, size must come
        # from the job's own [start, end) for that rank.
        job = done_job(client, world=3)
        jid = job["job_id"]
        r0 = job["ranks"][0]
        resp = plan(client, jid, [
            vol("vol-a", "rack-a", 0, [ref(jid, 0)]),
            vol("vol-b", "rack-b", 10**9),
        ], expandable="vol-b")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["placements"]) == 1
        assert body["placements"][0]["size_bytes"] == (r0["end"] - r0["start"]) * 4 * 2

    def test_already_redundant_shard_is_skipped(self, client):
        job = done_job(client)
        jid = job["job_id"]
        all_refs = [ref(jid, r) for r in range(4)]
        resp = plan(client, jid, [
            vol("vol-a", "rack-a", 0, all_refs),
            vol("vol-b", "rack-b", 0, all_refs),  # every shard on 2 racks already
            vol("vol-c", "rack-c", 10**9),
        ], expandable="vol-c")
        body = resp.json()
        assert body["feasible"] is True
        assert body["redundant_shards"] == 4
        assert body["backfill_shards"] == 0
        assert body["placements"] == []

    def test_two_replicas_on_same_rack_still_need_a_third(self, client):
        job = done_job(client)
        jid = job["job_id"]
        others = redundant_refs(job, keep={0})
        resp = plan(client, jid, [
            vol("vol-a1", "rack-a", 0, [ref(jid, 0)] + others),
            vol("vol-a2", "rack-a", 0, [ref(jid, 0)]),  # same rack: no help
            vol("vol-b", "rack-b", SHARD_BYTES, others),
        ], expandable="vol-b")
        body = resp.json()
        assert body["feasible"] is True
        assert len(body["placements"]) == 1
        assert body["placements"][0]["target_rack"] == "rack-b"

    def test_plan_is_side_effect_free(self, client):
        job = done_job(client)
        jid = job["job_id"]
        store = client.app.state.store
        before = (store.job_dir(jid) / "job.json").read_text()
        resp = plan(client, jid, [
            vol("vol-a", "rack-a", 0, [ref(jid, 0)]),
            vol("vol-b", "rack-b", SHARD_BYTES),
        ], expandable="vol-b")
        assert resp.status_code == 200
        assert (store.job_dir(jid) / "job.json").read_text() == before
        assert store.get(jid).status == "done"


class TestExpandableVolume:
    def test_growth_is_reported_when_others_are_full(self, client):
        job = done_job(client)
        jid = job["job_id"]
        others = redundant_refs(job, keep={0})
        resp = plan(client, jid, [
            vol("vol-a", "rack-a", 0, [ref(jid, 0)] + others),
            vol("vol-b", "rack-b", 1000, others),      # too small, not expandable
            vol("vol-c", "rack-c", 500),               # expandable
        ], expandable="vol-c")
        body = resp.json()
        assert body["feasible"] is True
        (p,) = body["placements"]
        assert p["target_volume"] == "vol-c"
        # 2000 bytes land on vol-c which has 500 free -> must grow by 1500.
        assert body["min_expand_bytes"] == SHARD_BYTES - 500
        vc = next(v for v in body["volumes"] if v["volume_id"] == "vol-c")
        assert vc["min_grow_bytes"] == SHARD_BYTES - 500
        assert vc["remaining_bytes"] == 500 - SHARD_BYTES
        # The non-expandable volume stays untouched within its budget.
        vb = next(v for v in body["volumes"] if v["volume_id"] == "vol-b")
        assert vb["planned_bytes"] == 0
        assert vb["min_grow_bytes"] == 0

    def test_expandable_free_bytes_are_used_before_growing(self, client):
        job = done_job(client)
        jid = job["job_id"]
        resp = plan(client, jid, [
            vol("vol-a", "rack-a", 0, [ref(jid, 0)]),
            vol("vol-c", "rack-c", SHARD_BYTES),  # expandable, exactly enough
        ], expandable="vol-c")
        body = resp.json()
        assert body["min_expand_bytes"] == 0
        vc = next(v for v in body["volumes"] if v["volume_id"] == "vol-c")
        assert vc["planned_bytes"] == SHARD_BYTES
        assert vc["remaining_bytes"] == 0


class TestContention:
    """Several shards competing for the same volume space."""

    def test_spill_to_expandable_when_shared_space_runs_out(self, client):
        job = done_job(client)
        jid = job["job_id"]
        others = redundant_refs(job, keep={0, 1})
        # ranks 0,1 only on rack-a; rack-b fits exactly one of them.
        resp = plan(client, jid, [
            vol("vol-a", "rack-a", 0, [ref(jid, 0), ref(jid, 1)] + others),
            vol("vol-b", "rack-b", SHARD_BYTES + 500, others),
            vol("vol-c", "rack-c", 0),  # expandable
        ], expandable="vol-c")
        body = resp.json()
        assert body["feasible"] is True
        assert len(body["placements"]) == 2
        targets = {p["rank"]: p["target_volume"] for p in body["placements"]}
        # One shard packs into vol-b, the other spills to the expandable vol-c.
        assert sorted(targets.values()) == ["vol-b", "vol-c"]
        vb = next(v for v in body["volumes"] if v["volume_id"] == "vol-b")
        assert vb["planned_bytes"] == SHARD_BYTES          # within its 2500
        assert vb["remaining_bytes"] == 500
        assert body["min_expand_bytes"] == SHARD_BYTES     # vol-c had 0 free

    def test_exact_packing_beats_greedy(self, tmp_path):
        # Uneven shard sizes: TOTAL=10, world=4 -> elements 2,3,2,3 ->
        # shard bytes 16,24,16,24.  A first-fit decreasing greedy would put
        # the 24 into vol-b (leaving 8) and be forced to grow the expandable
        # volume by 8; the exact search packs 24->vol-c and 16+16->vol-b and
        # needs zero growth.
        source = tmp_path / "src-small"
        write_source(source, total=10, step=1,
                     weight_bounds=[0, 10], momentum_bounds=[0, 10])
        app = create_app(Settings(source_dir=source, output_root=tmp_path / "out-small",
                                  sync_execution=True))
        with TestClient(app) as c:
            job = done_job(c, world=4)
            jid = job["job_id"]
            sizes = {r["rank"]: (r["end"] - r["start"]) * 8 for r in job["ranks"]}
            assert sizes == {0: 16, 1: 24, 2: 16, 3: 24}
            resp = plan(c, jid, [
                # rank 3 already redundant; ranks 0,1,2 need a cross-rack copy.
                vol("vol-a", "rack-a", 0, [ref(jid, 0), ref(jid, 1), ref(jid, 2)]),
                vol("vol-x", "rack-x", 0, [ref(jid, 3)]),
                vol("vol-y", "rack-y", 0, [ref(jid, 3)]),
                vol("vol-b", "rack-b", 32),
                vol("vol-c", "rack-b", 24),  # expandable
            ], expandable="vol-c")
            body = resp.json()
            assert body["feasible"] is True
            assert body["min_expand_bytes"] == 0
            targets = {p["rank"]: p["target_volume"] for p in body["placements"]}
            assert targets == {0: "vol-b", 1: "vol-c", 2: "vol-b"}

    def test_unresolvable_contention_reports_reason(self, client):
        job = done_job(client)
        jid = job["job_id"]
        others = redundant_refs(job, keep={0, 1})
        # Two 2000-byte shards, only 2500 bytes on the only other rack, and
        # the expandable volume sits on the source rack (not eligible).
        resp = plan(client, jid, [
            vol("vol-a", "rack-a", 0, [ref(jid, 0), ref(jid, 1)] + others),
            vol("vol-b", "rack-b", SHARD_BYTES + 500, others),
            vol("vol-a2", "rack-a", 10**9),  # expandable but same rack
        ], expandable="vol-a2")
        body = resp.json()
        assert body["feasible"] is False
        assert body["placements"] == []
        assert len(body["unplaced"]) == 2
        for u in body["unplaced"]:
            assert u["code"] == "capacity_contention"
            assert "contention" in u["reason"]


class TestUnplaceable:
    def test_no_source_replica(self, client):
        job = done_job(client)
        jid = job["job_id"]
        others = redundant_refs(job, keep={0, 1})
        # rank 0 is on no volume at all; rank 1 is fine and gets planned.
        resp = plan(client, jid, [
            vol("vol-a", "rack-a", 0, [ref(jid, 1)] + others),
            vol("vol-b", "rack-b", SHARD_BYTES, others),
        ], expandable="vol-b")
        body = resp.json()
        assert body["feasible"] is False
        assert len(body["placements"]) == 1  # the placeable shard still plans
        assert body["placements"][0]["rank"] == 1
        (u,) = body["unplaced"]
        assert u["rank"] == 0
        assert u["code"] == "no_source_replica"
        assert "no source replica" in u["reason"]

    def test_no_volume_on_other_rack(self, client):
        job = done_job(client)
        jid = job["job_id"]
        # Every volume sits on rack-a, so no shard can get a cross-rack copy.
        resp = plan(client, jid, [
            vol("vol-a", "rack-a", 10**9, [ref(jid, r) for r in range(4)]),
        ], expandable="vol-a")
        body = resp.json()
        assert body["feasible"] is False
        assert body["placements"] == []
        assert len(body["unplaced"]) == 4
        for u in body["unplaced"]:
            assert u["code"] == "no_eligible_rack"
            assert "different rack" in u["reason"]

    def test_shard_larger_than_any_eligible_volume(self, client):
        job = done_job(client)
        jid = job["job_id"]
        others = redundant_refs(job, keep={0})
        resp = plan(client, jid, [
            vol("vol-a", "rack-a", 0, [ref(jid, 0)] + others),
            vol("vol-b", "rack-b", SHARD_BYTES - 1, others),  # 1 byte short
            vol("vol-a2", "rack-a", 10**9),                   # expandable, wrong rack
        ], expandable="vol-a2")
        body = resp.json()
        assert body["feasible"] is False
        (u,) = body["unplaced"]
        assert u["code"] == "insufficient_capacity"


class TestLimitsAndValidation:
    def test_too_many_volumes_rejected(self, client):
        job = done_job(client)
        volumes = [vol(f"vol-{i}", f"rack-{i}", 0) for i in range(6)]
        resp = plan(client, job["job_id"], volumes, expandable="vol-0")
        assert resp.status_code == 422

    def test_too_many_backfill_shards_rejected(self, client):
        job = done_job(client, world=12)  # 12 shards, all single-rack
        jid = job["job_id"]
        resp = plan(client, jid, [
            vol("vol-a", "rack-a", 0, [ref(jid, r) for r in range(12)]),
            vol("vol-b", "rack-b", 10**9),
        ], expandable="vol-b")
        assert resp.status_code == 422
        assert "limit of 10" in resp.json()["detail"]

    def test_exactly_ten_backfill_shards_ok(self, client):
        job = done_job(client, world=10)
        jid = job["job_id"]
        resp = plan(client, jid, [
            vol("vol-a", "rack-a", 0, [ref(jid, r) for r in range(10)]),
            vol("vol-b", "rack-b", 10**9),
        ], expandable="vol-b")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["feasible"] is True
        assert body["backfill_shards"] == 10
        assert len(body["placements"]) == 10

    def test_unknown_job_404(self, client):
        resp = plan(client, "nope", [vol("vol-a", "rack-a", 0)], expandable="vol-a")
        assert resp.status_code == 404

    def test_unfinished_job_rejected(self, client):
        job = done_job(client)
        jid = job["job_id"]
        store = client.app.state.store
        stored = store.get(jid)
        stored.status = "failed"
        stored.error = "boom"
        store.save(stored)
        resp = plan(client, jid, [vol("vol-a", "rack-a", 0)], expandable="vol-a")
        assert resp.status_code == 422
        assert "completed" in resp.json()["detail"]

    def test_expandable_volume_must_exist(self, client):
        job = done_job(client)
        resp = plan(client, job["job_id"],
                    [vol("vol-a", "rack-a", 0)], expandable="vol-ghost")
        assert resp.status_code == 422
        assert "expandable_volume" in resp.json()["detail"]

    def test_duplicate_volume_id_rejected(self, client):
        job = done_job(client)
        resp = plan(client, job["job_id"], [
            vol("vol-a", "rack-a", 0),
            vol("vol-a", "rack-b", 0),
        ], expandable="vol-a")
        assert resp.status_code == 422
        assert "duplicate" in resp.json()["detail"]

    def test_shard_ref_outside_selected_jobs_rejected(self, client):
        job = done_job(client, world=4)
        other = done_job(client, world=5)
        resp = plan(client, job["job_id"], [
            vol("vol-a", "rack-a", 0, [ref(other["job_id"], 0)]),
        ], expandable="vol-a")
        assert resp.status_code == 422
        assert "not part of this plan" in resp.json()["detail"]

    def test_shard_ref_rank_out_of_range_rejected(self, client):
        job = done_job(client)
        jid = job["job_id"]
        resp = plan(client, jid, [
            vol("vol-a", "rack-a", 0, [ref(jid, 99)]),
        ], expandable="vol-a")
        assert resp.status_code == 422
        assert "ranks 0..3" in resp.json()["detail"]

    def test_duplicate_shard_on_same_volume_rejected(self, client):
        job = done_job(client)
        jid = job["job_id"]
        resp = plan(client, jid, [
            vol("vol-a", "rack-a", 0, [ref(jid, 0), ref(jid, 0)]),
        ], expandable="vol-a")
        assert resp.status_code == 422
        assert "twice" in resp.json()["detail"]


class TestMultipleJobs:
    def test_one_plan_can_cover_several_jobs(self, client):
        job_a = done_job(client, world=2)
        job_b = done_job(client, world=3)
        ja, jb = job_a["job_id"], job_b["job_id"]
        others = redundant_refs(job_a, keep={0}) + redundant_refs(job_b, keep={2})
        resp = plan(client, [ja, jb], [
            vol("vol-a", "rack-a", 0, [ref(ja, 0), ref(jb, 2)] + others),
            vol("vol-b", "rack-b", 10**9, others),
        ], expandable="vol-b")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["job_ids"] == [ja, jb]
        assert body["total_shards"] == 2 + 3
        assert body["feasible"] is True
        got = {(p["job_id"], p["rank"]) for p in body["placements"]}
        assert got == {(ja, 0), (jb, 2)}
