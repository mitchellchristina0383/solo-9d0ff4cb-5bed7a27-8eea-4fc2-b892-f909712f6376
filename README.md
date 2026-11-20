# Checkpoint Reshard API

训练进程数变化时，把同一份 float32 检查点（**权重 + 动量**）从只读源卷重切到新的进程数，
输出每个进程的全局下标区间和原始步数，供训练作业直接续跑。

- 源清单（`manifest.json`）声明步数、总长度、每个 float32 文件覆盖的全局下标区间；
  权重与动量的切分点**可以不同**。
- 源区间有**缺口或重叠**（或文件大小不符）→ 拒绝转换（HTTP 422）。
- 按新进程数把全局下标 `[0, total_length)` 均分为连续区间（大小差 ≤ 1 个元素），
  分块流式读写（默认 4 MiB，可用 `CHUNK_BYTES` 调整）。
- 每个 rank 目录同时产出该区间的 `weights.f32` 和 `momentum.f32`，外加
  `shard.json`（区间 + 原始步数）。
- 中断/重启后续做：**已完整落盘的分片原样复用**，只补未完成的；
  **整组齐备前下载一律 409**。

## 快速开始（Docker Compose）

仓库自带一份本地样例（`sample/source/`，步数 7459、总长 10007、权重 3 段 / 动量 3 段不同切分）：

```bash
docker compose up --build
# 打开 http://localhost:8000/docs
```

如需重新生成样例：`python3 sample/make_sample.py sample/source`

不用 Docker 本地跑：

```bash
pip install -r requirements.txt
SOURCE_DIR=$PWD/sample/source OUTPUT_ROOT=$PWD/outputs \
  uvicorn app.main:create_app --factory --port 8000
```

## API（均可在 /docs 中操作）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/reshard` | 发起转换：`{"new_world_size": 6}`；相同请求幂等返回同一任务 |
| GET | `/jobs` / `/jobs/{job_id}` | 任务列表 / 进度（含各 rank 区间、步数、完成数） |
| POST | `/jobs/{job_id}/resume` | 失败/重启后续做；已完成分片自动跳过 |
| GET | `/jobs/{job_id}/shards/{rank}/meta` | 该 rank 的区间 `[start, end)` 与原始步数 |
| GET | `/jobs/{job_id}/shards/{rank}/download?part=weights\|momentum` | 下载分片；任务未全部完成时 409 |
| POST | `/backup/plan` | 备份扩容试算：给已完成任务的分片规划跨机架副本（只出计划） |

```bash
# 发起（source_dir 缺省用挂载的 SOURCE_DIR，也可显式覆盖）
curl -X POST localhost:8000/reshard -H 'Content-Type: application/json' \
     -d '{"new_world_size": 6}'
# 查询 → 拿到 job_id 与每个 rank 的 [start, end) 和 step
curl localhost:8000/jobs/<job_id>
# 中断后续做
curl -X POST localhost:8000/jobs/<job_id>/resume
# 整组完成后下载
curl -OJ "localhost:8000/jobs/<job_id>/shards/0/download?part=weights"
```

## 备份扩容试算（`POST /backup/plan`，只出计划）

检查点按分片分开存放；整架停机时也要能取回全部分片，因此每个分片都需要
**跨机架副本**。管理员选一批**已完成**的重分片任务，上报各备份卷的所属机架、
剩余字节和已有的完整分片，并指定**唯一可扩容**的卷，服务试算一份扩容计划：

- 只在**一个机架**上有副本的分片 → 各补一份到别的机架，响应给出每份新副本的
  源卷和目标卷（`placements[]`）；
- 可扩容卷**至少要增加的字节数**（`min_expand_bytes`，已扣除其现有余量）；
  其他卷绝不超出余量（`volumes[].remaining_bytes ≥ 0`）；
- 排不了的分片在 `unplaced[]` 里给出原因（`code`）：
  - `no_source_replica`：提交的卷都不持有该分片，没有可复制的源；
  - `no_eligible_rack`：所有卷都和现有副本同机架；
  - `insufficient_capacity`：别的机架上没有装得下它的卷，可扩容卷又同架；
  - `capacity_contention`：单个都放得下，但整批合计超过别的机架的总余量
    （此时整组待补分片都不落位，不替管理员挑"谁先谁后"）。

规则与限制：

- 分片大小按任务区间计算：`(end - start) × 4 字节 × 2` —— **权重和动量同卷
  存放**，一个副本两份一起算；
- 已在 ≥2 个机架有副本的分片自动跳过（计入 `redundant_shards`）；
- 每次最多试算 **10 个待补分片**、**5 个卷**，超出返回 422；
- 放置用精确搜索（不是贪心）：多个分片争用卷空间时，给出的可扩容卷增加字节数
  是最小的；
- 只出计划：不复制数据、不改任务状态。卷的"剩余字节"和"已有分片"由管理员如实
  上报，服务只校验分片属于所选任务，无法核实远端卷的真实内容。

### /docs 演示（Docker Compose）

`docker compose up --build` 之后，全部可以在 http://localhost:8000/docs 点出来：

1. `POST /reshard` 提交 `{"new_world_size": 6}`。样例 + 默认挂载下任务 id 固定为
   `94dd65ae4907a5c3`（改过样例或挂载路径则以 `GET /jobs` 实际返回为准），
   轮询到 `status == "done"`。
2. `POST /backup/plan` —— /docs 的示例请求体就是下面这份（把
   `<job_id from POST /reshard>` 全部换成上一步的 id）：6 个分片都在 rack-a 的
   `vol-a1` 上，rack-b 的 `vol-b1` 余量 30000，rack-c 的 `vol-c1` 余量 8000
   且是唯一可扩容卷。

```json
{
  "job_ids": ["94dd65ae4907a5c3"],
  "volumes": [
    {"volume_id": "vol-a1", "rack": "rack-a", "free_bytes": 20000,
     "shards": [{"job_id": "94dd65ae4907a5c3", "rank": 0},
                {"job_id": "94dd65ae4907a5c3", "rank": 1},
                {"job_id": "94dd65ae4907a5c3", "rank": 2},
                {"job_id": "94dd65ae4907a5c3", "rank": 3},
                {"job_id": "94dd65ae4907a5c3", "rank": 4},
                {"job_id": "94dd65ae4907a5c3", "rank": 5}]},
    {"volume_id": "vol-b1", "rack": "rack-b", "free_bytes": 30000, "shards": []},
    {"volume_id": "vol-c1", "rack": "rack-c", "free_bytes": 8000,  "shards": []}
  ],
  "expandable_volume": "vol-c1"
}
```

3. 响应（要点）：每个分片 13336~13344 字节（区间约 1668 元素 × 4 × 2）；
   rank 1、2 落到 `vol-b1`（用掉 26688，余 3312，不超余量），rank 0、3、4、5
   落到 `vol-c1`（共 53368），`min_expand_bytes = 45368` —— 即 `vol-c1` 在现有
   8000 余量之外至少还要扩这么多。把 `vol-c1` 的 `free_bytes` 改大再试，
   `min_expand_bytes` 会相应减小，直至 0。

## 源清单格式

```json
{
  "step": 7459,
  "total_length": 10007,
  "weights":  [{"path": "weights/part-0.f32",  "start": 0,    "end": 3500}, ...],
  "momentum": [{"path": "momentum/part-0.f32", "start": 0,    "end": 2000}, ...]
}
```

- 区间为**左闭右开**的全局元素下标；每类的所有区间必须**无缝无重叠**地铺满
  `[0, total_length)`，否则 422。
- 每个文件必须是**原始 little-endian float32**，大小恰好 `(end-start)*4` 字节。
- `path` 相对源目录，拒绝绝对路径与 `..`。

## 输出布局

```
outputs/<job_id>/
  job.json                 # 任务状态，每完成一个分片落盘一次
  rank_00000/
    weights.f32            # 区间 [start, end) 的权重
    momentum.f32           # 同一区间的动量
    shard.json             # {"rank","start","end","step",...} —— 最后写入的完成标记
```

## 设计要点

- **均分**：rank `i` 得 `[i*N//W, (i+1)*N//W)`，连续、全覆盖、大小差 ≤ 1。
- **崩溃安全**：数据先写 `*.tmp` 再 `os.replace` 原子改名；`shard.json` 最后写，
  作为分片完成标记；`job.json` 同样原子替换。服务重启时把 `running` 任务标记为
  可续做的 `failed`。
- **复用判定**：分片目录里标记文件存在且两个数据文件大小精确匹配 → 视为完整，
  续做时直接跳过（不重写）。
- **源变更防护**：续做前重新校验清单，步数/总长/区间与任务不一致则失败。
- **下载闸门**：仅当任务状态为 `done`（全部 rank 齐备）才放行下载，否则 409。
- **幂等任务**：`job_id` 由（源目录、清单内容、新进程数）哈希得出，重复发起
  同一请求返回同一任务。

## 测试

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

覆盖：均分数学、清单校验（缺口/重叠/大小/路径穿越）、端到端字节级一致性、
中断续做（完整分片不重写）、重启后复用、源变更检测、下载闸门、API 全流程，
以及备份试算（跨机架放置、可扩容卷增长、多分片争用卷空间、10 分片/5 卷上限、
输入校验）。

## 已知限制

- 单进程服务：任务以线程方式跑在一个 uvicorn worker 内，请勿开多 worker。
- 下载按文件提供（weights/momentum 分开），未提供打包归档接口。
- 源文件在任务创建后被**同尺寸同步数**篡改无法识别（未做内容校验和）。
- `new_world_size > total_length` 会被拒绝（每个进程至少分到一个元素）。
- 备份试算只出计划、不执行复制；计划的正确性取决于管理员上报的卷余量与
  已有分片是否属实（服务无法核实远端备份卷）。
