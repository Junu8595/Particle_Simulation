# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Graph Neural Network-based DEM (Discrete Element Method) particle simulator. Predicts particle dynamics in industrial equipment (hoppers, rollers) by learning particle interactions through graph message-passing networks with local coordinate frame invariance. Based on the scalarization-vectorization paradigm from the DYNAMI-CAL GRAPHNET paper (Nature Communications, 2026).

## Environment (Fixed — Do Not Re-discover)

### Local Environment (Windows + WSL Ubuntu)
- Project path: /mnt/c/Users/AISDL_PJW/Projects/Particle_Simulation/
- Python: /mnt/c/Users/AISDL_PJW/AppData/Local/Programs/Python/Python312/python.exe
- Git branch: main
- GPU: RTX 4070 Super (CUDA available)

### Remote Environment (Linux GPU Server)
- SSH: ssdl@147.47.206.229
- Project path: /home/ssdl/PJW/Particle_Simulation/
- Conda env: PJW
- GPU: Ada 6000 (use CUDA_VISIBLE_DEVICES=0, shared with 3 colleagues)

### Dependencies (All Pre-installed)
All dependencies listed in "Dependencies" section are already installed in both environments.
Do NOT run pip/conda install. If an import fails, report it and stop.

### Startup Rules for Claude Code
1. Do NOT run environment discovery commands (which python, ls ~/.claude, pwd, etc.) — paths are documented above.
2. Do NOT install packages — they already exist.
3. Do NOT re-read large .npy/.pt files (>10MB) — use Python scripts to extract specific metrics.
4. For pred.npy/targ.npy analysis, use the diagnostic script template in @debugger agent.
5. Start requested tasks immediately without preliminary exploration.

## Running the Code

All configuration is in `attributes.py` — edit that file first to set data paths and hyperparameters.

**Train:**
In `graph_main.py`, set `train_flag = True`, `fresh_start = True` (new run) or `False` (resume checkpoint), then:
```bash
python graph_main.py
```

**Inference / Rollout:**
Set `train_flag = False`, point `test_network_path` to a checkpoint directory, set `roll_out_flag = True` for multi-step prediction, then run `python graph_main.py`.

**Key flags at top of `graph_main.py`:**
- `train_flag` — True = train, False = inference
- `roll_out_flag` — True = auto-regressive multi-step rollout
- `fresh_start` — True = train from scratch, False = load checkpoint
- `plot_flag` — True = render video outputs
- `one_step_flag` — True = single-step prediction only

**Data preprocessing (bakes expensive graph ops to disk):**
```bash
python preprocess_data.py       # bake RawDataPack → DataPack as .pt files
python check_data.py            # validate baked data format
```

**Post-processing analysis (RMSE, momentum conservation):**
```bash
python post_processing.py
```

**`0406_Baseline/`** mirrors the root-level files and represents a prior stable experiment snapshot.

## Architecture & Data Flow

### Pipeline

```
.npy files → dataset.py → graph_builder.py → graph_model.py → loss → optimizer
```

1. **Data Loading** (`dataset.py`): Loads time-series `.npy` files (particle positions, mesh geometry, node types). Constructs sliding windows of 5 history frames. Returns `RawDataPack`.

2. **Graph Construction** (`graph_builder.py`):
   - Particle-particle (PP) edges: `radius_graph` search (distance ≤ `contact_distance`)
   - Particle-mesh (PM) edges: `build_boundary_edge()` — closest point on triangle mesh, with face normal
   - PP edges get antisymmetric local frame `(a_ij, b_ij, c_ij)` via `build_edge_local_frame_3d()`

3. **Feature Engineering** (`dataset.py` `graph_data()`):
   - Node features: velocity history (15D) + one-hot node type (4D) + surface normal (6D) = 25D
   - Edge features: distance (1D) + local-frame-projected relative position (3D) + relative velocity (3D) = 7D
   - Target: acceleration, normalized then log-transformed: `log(|x|+1) * sign(x)`
   - All features normalized via `normalizer.py` (online running statistics)

4. **GNN Forward Pass** (`graph_model.py` → `graph_networks.py`):
   - **① Encoder**: 25D→128D nodes, 7D→128D edges (MLPs with LayerNorm)
   - **② Processor**: 10 message-passing steps with attention + residual connections
   - **③ Decoder** — 두 경로의 합산:

   **■ Edge Force Decoder (Pairwise Interaction)**
   - Scalar coefficient 출력: (E, 128) → (E, 3) — PP/PM decoder 분리 (pairwise_mask)
   - `f_ij = s1·a + s2·b + s3·c` vectorization (edge local frame 기반)
   - `scatter_add(f_ij, receivers)` → Δv_internal = Σ f_ij  *(Newton 제3법칙 보존)*

   **■ External Force Decoder (Non-pairwise Force)**  ← 신규 (`node_decoder`)
   - Node latent에서 직접 출력: (N, 128) → (N, 3)
   - 중력 등 외력 + edge로 표현 불가한 잔여 성분 보정
   - → Δv_ext = ψ(h_i)  *(논문 Fig.1c)*

   **■ 최종 출력**: `output = edge_output + node_residual` → Δv_net = Δv_internal + Δv_ext

5. **Loss**: `normalized_target - output` on particle nodes only. Relative RMSE: `sqrt(MSE) / sum(target²)` (Classic GNN 방식) — Issue #7 조사 중.

6. **Optimization**: Adam, exponential + linear LR decay, gradient clipping (`max_norm=3.0`).

### Inference Modes

- **Single-step** (`test_cycle`): One prediction per call, GT positions supplied
- **Multi-step rollout** (`test_cycle` with state carried over): Iteratively feeds predictions back
- **Spatial tiling** (`grid_test_cycle`): Recursively partitions domain into overlapping boxes, runs inference per tile, blends predictions — used for large domains

## Module Reference

| File | Role |
|---|---|
| `attributes.py` | **Single source of truth** for all hyperparameters, network dims, data paths |
| `graph_main.py` | Entry point; training loop with DataLoader; test cycles; spatial tiling; `gns_collate_fn()` |
| `graph_model.py` | Graph wrapper: encode→process→decode; Edge Force Decoder (PP/PM) + External Force Decoder (`node_decoder`); loss |
| `graph_networks.py` | MLP building blocks; `graph_net` module with `sub_nets` ModuleList |
| `graph_builder.py` | `build_edge_local_frame_3d()`, `build_boundary_edge()`, `safe_normalize()`, `build_fallback_b_from_a()` |
| `dataset.py` | Data loading; `RawDataPack → DataPack`; `graph_data()` feature engineering; `__getitem__()` for baked data |
| `normalizer.py` | Online statistics accumulation + z-score normalization; `inverse()` for denormalization |
| `preprocess_data.py` | Offline dataset baking: calls `graph_data()` and saves DataPack as `.pt` files |
| `post_processing.py` | RMSE/momentum analysis; comparison plots |
| `graph_utils.py` | Logging, LR scheduling, code backup utilities |

## Configuration (`attributes.py`)

- `latent_size = 128`, `message_passing_steps = 10`
- `contact_distance = 0.75` (radius for edge creation, in mm after ×1000 scaling)
- `nepochs = 500`, `lr = 1e-4`, `num_history = 5`
- Node encoder input = 25D, Edge encoder input = 7D, Decoder output = 3D
- Dataset: `training_40_/` folder with 40 simulation sequences as `.npy` files
- Spatial domain: X/Y ∈ [-50, 50] mm, Z ∈ [0, 100] mm (raw data is in meters, scaled ×1000)
- Node types: `particle=0`, `hopper=1`, `roller1=2`, `roller2=3`

## Dependencies

PyTorch, PyTorch Geometric (`radius_graph`), PyTorch Scatter (`scatter_add`, `scatter_softmax`), scikit-learn, matplotlib, OpenCV.

---

## Known Issues & Fix History

---

## ✅ Issues #1–#6 Resolved (as of 2026-04-16)

Re-baked data (6176 `.pt` files) uploaded to remote server.

| Issue | Commit | Summary |
|-------|--------|---------|
| #1 Contact edge local frame = 0 | `417de37` | Gram-Schmidt frame (a,b,c) built for PM edges via mesh face normal |
| #2 Contact edge feature 0-padding | `417de37` | dx_local/dv_local projected for contact edges via `cm = ~pairwise_mask` |
| #3 Normalizer baking mismatch | `417de37` | `bake_mode=True` stores raw features; normalization applied in `__getitem__()` |
| #4 collate_fn particle_indices offset | `80e3bc8` | particle_indices / next_particle_indices accumulated with node_offset per batch element |
| #5 Log/Exp numerical instability | `80e3bc8` | clamp(max=10.0) added in `reverse_output()` before exp |
| #6 PP local frame b ∦ a | `192bed3` | velocity-based b'_ij (DYNAMI-CAL 방식) + Gram-Schmidt + cross(b_perp, a) |

**Validation on real step_0 data (2026-04-15):**
- PP edges: 144,161 / PM edges: 15,982
- PP `max |a·b| = 0.000000` ✅ (이전: 1.000000)
- PM `max |a·b| = 0.000250` ✅
- All frame vectors unit-norm, b_degenerate = 0

---

## preprocess_data.py — GPU 가속 (commit `b4dc180`, 2026-04-15)

GPU 자동 사용 (`self.device = cuda:0`). `searchsorted` 벡터화 + 텐서 인덱싱으로 ~4s/step → 0.41s/step (**10x**), 6176 steps ≈ 40분.

```bash
python preprocess_data.py
```

---

## 🔄 Issue #7 — Y-axis rollout divergence (조사 진행 중)

**증상:** t=48에서 Y-RMSE 683mm 폭발, t=192에서 ~3616mm. Acc std ratio @ t=48: X≈0.18, Y≈3.87, Z≈0.37. 40k/100k/260k step 동일 → stagnation.

**조사 이력:**

| 시도 | 공식 | 결과 |
|------|------|------|
| 1: plain MSE | `error².sum(dim=1).mean()` | X=0.05, Y=2.26, Z=0.24 @ 100k |
| 2: per-axis MSE | `error².mean(dim=0).mean()` | plain MSE/3과 동일, 효과 없음 |
| 3: relative RMSE (현재) | `sqrt(MSE)/sum(target²)` | X=0.183, Y=3.871, Z=0.375 @ 260k, 변화 없음 |

**핵심 발견 (2026-04-19):**
- Normalizer 정상 ✅ — z-score 후 Y/X/Z std 비율 = 1.000 (loss·normalizer는 원인 아님)
- Y-bias 구조적 원인: 중력 적층으로 PP 엣지 `a` 벡터가 Y 방향 편중 → `s1·a`가 Y force 지배
- **대응:** External Force Decoder (`node_decoder`) 추가로 edge 독립적 보정 경로 확보 (commit `b1dcd03`)

**현재 코드:** relative RMSE + 매 10k step `[DEBUG]` 로깅 (output/node_residual/f_ij 축별 stats)

**결론:** node_decoder 추가 자체는 유효한 방향이나, edge decoder 축퇴(degeneration) 문제가 새로운 이슈로 부상 → Issue #8 참조.

---

## 🔄 Issue #8 — Edge Decoder 축퇴 (비교실험 진행 중, 2026-04-20~)

### 문제 정의

antisymmetric edge decoder(`f_ij = s1·a + s2·b + s3·c`)와 external force node decoder를 혼합 운용할 때, **edge decoder가 학습을 포기하고 node decoder에 수렴을 위임하는 축퇴 현상** 발생.

- 증상: 40k step 이후 Abs Loss ≈ 0.034~0.041에서 plateau, `[DEBUG]` edge f_ij std ≈ 0.001 (node_residual std ≈ 0.05 대비 50배 작음)
- 원인 가설: node decoder의 gradient magnitude가 edge decoder보다 커서 optimizer가 node decoder를 주학습 경로로 선택

### 비교실험 설계

**기준 실험 (Issue #7 말기):** batch=4, relative RMSE, edge+node decoder, 260k step → stagnation

**조정 대상 하이퍼파라미터:**

| 변수 | 설명 | 예상 효과 |
|------|------|-----------|
| `batch_size` | 배치 크기 축소 (4→1) | gradient noise 증가 → 지역 최솟값 탈출 가능성 |
| LR decay rate / 하한 | decay 저감으로 후반 lr 유지 | 장기 학습 시 gradient 소실 방지 |
| Phase step | edge-only 학습 후 node decoder 개입 타이밍 | edge decoder가 먼저 수렴한 뒤 node가 잔차 보정 |

### 실험 목록

| 실험 | 구조 | batch | loss | 비고 |
|------|------|-------|------|------|
| A (기준) | edge-only | 1 | rel RMSE | 260k 실험과 동일 구조, batch만 1로 변경. 100k step 후 비교 |
| B | edge-only | 1 | rel RMSE | LR decay 조정 |
| C | edge + node (phase) | 1 | rel RMSE | phase step 조정 |

### 현재 코드 상태 (2026-04-20)

- `batch_size = 1`, edge-only decoder (node_decoder 아키텍처에서 제거)
- `output = edge_output` (node_residual 없음)
- `[DEBUG]` 로그 → log 파일 기록 (`set_log_path` 유지)

### 실험 결과 (2026-04-20~21)

| 실험 | 설명 | 100k Abs Loss (최저) | output Y/X @ 100k |
|------|------|------|------|
| Exp A | edge-only, batch=1, secondary_decay=1e-2 | 0.0338 @ 90k | 17% |
| Exp B | edge-only, batch=1, secondary_decay=1e-1 | 0.0340 @ 90k | 16% |

- Exp A와 Exp B 수렴 곡선이 100k 구간에서 사실상 동일 → `secondary_decay_offset` 효과는 LR이 Phase 2(800k step 이후)에 진입해야 나타남
- output Y/X 15~18% 정체: loss/LR이 아닌 **PM frame 구조적 Y 결핍** 문제 → Issue #9 참조

---

## 🔄 Issue #9 — PM edge_a Y-결핍 (hopper geometry bias, 2026-04-21)

### 문제 정의

output Y/X 비율이 15~18%로 학습 내내 개선되지 않는 근본 원인 규명.
`baked_training_data/step_0.pt` 분석 결과, **PM edge local frame의 구조적 Y 결핍** 확인.

### 데이터 근거 (`step_0.pt` 기준)

| | PP (144,149개) | PM (15,984개) |
|---|---|---|
| edge_a \|Y\| / \|X\| | **1.04** ✅ | **0.61** ⚠️ |
| edge_b \|Y\| / \|X\| | 0.99 ✅ | 0.93 ✅ |
| edge_c \|Y\| / \|X\| | 0.98 ✅ | 1.46 |

- **PP**: a/b/c 모두 X/Y/Z 균등 분포 → PP f_ij Y/X 학습 중 87~102% 유지 (정상)
- **PM edge_a**: Y 성분이 X의 61%에 불과 → `s1·a` (PM 주력 decoder 성분)가 Y force를 구조적으로 과소 표현

### 원인 분석

PM `edge_a` = 메시 삼각면의 면 법선(face normal). Hopper 형상 특성상:
- 수직 측벽(Z축 방향 법선) 비중이 높음 → `edge_a`의 Z 성분 과잉(|Z|=0.649, |Y|=0.256)
- `edge_c = a × b` 크로스곱 결과로 Y 과잉(1.46)이 나타나지만, decoder 기여도는 `s1·a` 위주
- PM f_ij Y/X가 학습 100k 내내 33~48%로 정체되는 이유

### 영향

- output 전체 Y/X = 15~18% 고착 (PP는 87~102%로 정상 → PM이 bottleneck)
- PM edges가 전체의 ~10%(15,984 / 160,133)이지만 boundary force를 담당 → rollout Y-divergence의 직접 원인 가능성

### 대응 방안 (미결)

1. **PM decoder 가중치 조정**: PM f_ij에서 `s3·c` 기여를 높이는 방향 (c는 Y 과잉으로 보완 가능)
2. **PM별 좌표계 재설계**: face normal 대신 particle-wall 상대벡터 기반으로 `edge_a` 재정의
3. **Y-weighted loss**: PM edges에 대해 Y축 loss 가중치 상향

---

## 🔄 Issue #10 — Gravity 제거 전처리 (2026-04-24)

### 문제 정의

target_acc의 Y축 전체 평균이 `-0.01150469`로, 중력 상수가 학습 타겟에 상수 bias로 포함되어 있음.
→ normalizer가 Y축 std를 **3.151**로 과대 추정 (X=0.039, Z=0.120 대비 80배) → normalized space에서 Y 신호 17배 축소 → Y 학습 불리.

### 근거

- target_normalizer Y std = **3.151** (X=0.039, Z=0.120)
- step_0 normalize 후 Y std = 0.0016 (X std = 0.028의 1/17)
- input 0.01 → reverse_output Y = **0.031** (X = 0.0004의 79배 증폭) → rollout 발산 직접 원인

### 수정 내용

| 파일 | 변경 |
|------|------|
| `attributes.py` | `gravity_y = -0.01150469` 상수 추가, `DataParameterPack`에 포함 |
| `dataset.py` `__init__` | `self.gravity_y` 저장 |
| `dataset.py` bake_mode | particle rows에서 `gravity_y` 제거 후 `.pt` 저장 |
| `dataset.py` `reverse_output` | `acc_prediction[:, 1] += self.gravity_y` (중력 복원) |

### gravity 추정 방법

전체 6176개 `.pt` 파일의 particle node `target_acc` Y축 mean:
```
gravity_y = mean(target_acc_particle[:, 1]) = -0.01150469
총 샘플: 11,158,631 particle-steps
X mean = -0.0000029 ≈ 0  ✅
Z mean = +0.0000005 ≈ 0  ✅
```

### 결과 (2026-04-26 완료)

re-bake 완료 (6176개, 2026-04-26 20:04~20:42). target_normalizer Y std 검증:

| 축 | 이전 std | 이후 std | 변화 |
|---|---|---|---|
| X | 0.039 | 0.0075 | — |
| Y | **3.151** | **0.0294** | **107배 감소** ✅ |
| Z | 0.120 | 0.0097 | — |

Y/X std 비율: 80.8× → 3.9× (잔류 비율은 호퍼 구조 원인 — Issue #9)

---

## ✅ Issue #11 — Domain-exit 입자 허수 target_acc 제거 (2026-04-26)

### 문제 정의

다음 timestep에서 도메인을 이탈하는 입자는 `pos[t+1] = (0,0,0)`으로 리셋.
이 때 `acc = 0 - 2·pos[t] + pos[t-1]` → Y-acc ≈ ±38mm (허수값).
→ Issue #10의 gravity 제거 후에도 target_normalizer Y std = 3.151의 **직접 원인**.

### 수정 내용

| 파일 | 변경 |
|------|------|
| `dataset.py` `NodePack` | `valid_particle_indices` 필드 추가 |
| `dataset.py` `graph_data()` | `valid_particle_indices = particle_indices[next_particle_id[particle_indices] > 0]` 생성 |
| `dataset.py` normalizer 누적 | `target_acc[valid_particle_indices]` 기준으로 통계 누적 (non-bake & `__getitem__` 모두) |
| `graph_model.py` encoder | `self.valid_particle_indices` 저장 |
| `graph_model.py` decoder loss | `next_particle_indices` → `valid_particle_indices` |
| `graph_main.py` collate_fn | `valid_particle_indices` 오프셋 적용 및 NodePack 포함 |

### 검증 결과 (`test_valid_particle_indices.py`)

- step 100 기준: particle 1615개 → valid 1597개 (domain-exit **18개** 제외)
- 이탈 입자 |Y-acc| mean=38.0mm vs 유효 입자 0.02mm (**1754배 차이**)
- re-bake 후 raw/baked 카운트 일치 ✅

### 최종 target_normalizer 수렴값

| 축 | mean | std |
|---|---|---|
| X | -0.000003 | 0.0075 |
| **Y** | -0.002604 | **0.0294** (이전 3.151 → **107배 감소**) |
| Z | 0.000000 | 0.0097 |

---

## 🔄 Issue #8 — Edge Decoder 축퇴 (Exp D 진행 중, 2026-04-27~)

### 문제 정의

antisymmetric edge decoder(`f_ij = s1·a + s2·b + s3·c`)와 external force node decoder를 혼합 운용할 때, **edge decoder가 학습을 포기하고 node decoder에 수렴을 위임하는 축퇴 현상** 발생.

- 증상: 40k step 이후 Abs Loss ≈ 0.034~0.041에서 plateau, `[DEBUG]` edge f_ij std ≈ 0.001 (node_residual std ≈ 0.05 대비 50배 작음)
- 원인 가설: node decoder의 gradient magnitude가 edge decoder보다 커서 optimizer가 node decoder를 주학습 경로로 선택

### 비교실험 설계

**기준 실험 (Issue #7 말기):** batch=4, relative RMSE, edge+node decoder, 260k step → stagnation

**조정 대상 하이퍼파라미터:**

| 변수 | 설명 | 예상 효과 |
|------|------|-----------|
| `batch_size` | 배치 크기 축소 (4→1) | gradient noise 증가 → 지역 최솟값 탈출 가능성 |
| LR decay rate / 하한 | decay 저감으로 후반 lr 유지 | 장기 학습 시 gradient 소실 방지 |
| Phase step | edge-only 학습 후 node decoder 개입 타이밍 | edge decoder가 먼저 수렴한 뒤 node가 잔차 보정 |
| node_residual penalty | node_decoder 독점 억제용 L2 패널티 | edge decoder gradient 공간 확보 |

### 실험 목록

| 실험 | 구조 | batch | loss | 비고 |
|------|------|-------|------|------|
| A (기준) | edge-only | 1 | rel RMSE | 100k Abs Loss 0.0338 @ 90k, Y/X 17% |
| B | edge-only | 1 | rel RMSE | LR decay 조정. 100k Abs Loss 0.0340 @ 90k, Y/X 16% |
| C | edge + node (phase 100k) | 1 | rel RMSE | 33k step 진행 중 (Abs Loss ~0.28) |
| **D** | **edge + node (동시)** | **1** | **rel RMSE + node penalty (λ=0.1)** | **2026-04-27 시작** |

### Exp D 코드 상태 (2026-04-27)

- phase gating 제거: `node_residual` 처음부터 활성화
- `output = edge_output + node_residual` (0.1 스케일링 제거)
- loss = `rel_RMSE + 0.1 × node_residual_L2`
- DEBUG 로그: `node/edge ratio` 추가 (ratio ≫ 1이면 node 독점 → λ 상향 필요)
- Issue #10+#11 수정 반영된 re-baked data 사용 (Y std=0.029)

### Exp A/B 결론

Exp A와 Exp B 수렴 곡선이 100k 구간에서 사실상 동일 → `secondary_decay_offset` 효과는 LR이 Phase 2(800k step 이후)에 진입해야 나타남.
output Y/X 15~18% 정체: **Issue #11 수정 전 데이터** 사용 → Y std 과대 추정이 원인이었음.
Exp D부터 Issue #10+#11 수정 re-baked data 적용.

---

## 🟡 Active Issues

Issue #8 Exp D — edge+node 동시학습 + node_penalty, 2026-04-27 시작.
Issue #9 PM frame Y-결핍 — 대응 방안 검토 중.

---

## Paper vs Current Code: Future Improvements (Non-blocking)

These differences from the DYNAMI-CAL GRAPHNET paper affect performance but do NOT block training:

1. ~~**b'_ij construction**: Paper uses velocity + angular_velocity~~ → **#6에서 velocity 기반으로 수정 완료**
2. **Spatiotemporal message passing**: Paper updates position/velocity via Euler integration at each MP step; current code only updates latent space
3. **Boundary handling**: Paper uses ghost node reflection (mesh-free); current code uses mesh triangulation + closest point projection
4. **Node-level decoding**: Paper decodes inverse mass/inertia from node embeddings; current code uses Edge Force Decoder + External Force Decoder (node_decoder, commit `b1dcd03`)
