# CLAUDE.md

Claude Code용 프로젝트 가이드 문서.

## Project Overview

Graph Neural Network 기반 DEM 입자 시뮬레이터. Hopper+Roller 장비에서 입자 거동 예측.
DYNAMI-CAL GRAPHNET 논문 (Nature Communications, 2026)의 scalarization-vectorization 구조 적용.

## Environment

### Local (Windows + WSL Ubuntu)
- Project path: /mnt/c/Users/AISDL_PJW/Projects/Particle_Simulation/
- Python: /mnt/c/Users/AISDL_PJW/AppData/Local/Programs/Python/Python312/python.exe
- Git branch: main
- GPU: RTX 4070 Super

### Remote (Linux GPU Server)
- SSH: `ssdl@147.47.206.229`
- Path: `/home/ssdl/PJW/Particle_Simulation/`
- Conda env: `PJW`
- GPU: Ada 6000 (`CUDA_VISIBLE_DEVICES=0`, 선배 3명과 공유)

### Startup Rules
1. Do NOT run environment discovery commands (which python, ls ~/.claude, pwd, etc.) — paths are documented above.
2. Do NOT install packages — they already exist.
3. Do NOT re-read large .npy/.pt files (>10MB) — use Python scripts to extract specific metrics.
4. For pred.npy/targ.npy analysis, use the diagnostic script template in @debugger agent.
5. Start requested tasks immediately without preliminary exploration.

### Dependencies (All Pre-installed)
All dependencies listed in "Dependencies" section are already installed in both environments.
Do NOT run pip/conda install. If an import fails, report it and stop.

---

## Running the Code

모든 설정은 `attributes.py`에서 관리.

```bash
# 학습
python graph_main.py          # train_flag=True, fresh_start=True/False

# Baking (무거운 그래프 연산 사전 계산)
python preprocess_data.py

# 후처리 분석
python post_processing.py
```

**`graph_main.py` 주요 플래그:**
- `train_flag` — True=학습, False=추론
- `roll_out_flag` — True=multi-step rollout
- `fresh_start` — True=처음부터, False=체크포인트 로드
- `one_step_flag` — True=single-step만

---

## Architecture

```
.npy → dataset.py → graph_builder.py → graph_model.py → loss → optimizer
```

### 노드/엣지 피처
- Node: velocity history (15D) + node type one-hot (4D) + surface normal (6D) = **25D**
- Edge: distance (1D) + local-frame dx (3D) + relative velocity (3D) = **7D**
- Target: acceleration → z-score → `log(|x|+1)·sign(x)`

### GNN 구조
- Encoder: 25D→128D (node), 7D→128D (edge)
- Processor: 10 message-passing steps (attention + residual)
- Decoder: 두 경로 합산

| 경로 | 구조 | 역할 | Newton 3법칙 |
|---|---|---|---|
| Edge Force Decoder | `f_ij = s1·a + s2·b + s3·c` → scatter_add | 입자간 접촉력 (antisymmetric) | ✅ 보장 |
| External Force Decoder | latent_node → MLP → node_residual | 중력/외력/자유낙하 (Classic GNN) | ❌ 없음 |

**최종 출력:** `output = edge_output + node_residual`

### Loss
```
error         = target - output
loss_main     = sqrt(MSE(error)) / sum(target²)       ← relative RMSE
node_penalty  = MSE(node_residual)                    ← node 독점 억제 (L2)
momentum_loss = MSE(mean(node_residual, dim=particles)) ← 운동량 편향 억제
loss_total    = loss_main + 0.1 × node_penalty + 0.01 × momentum_loss
```

### Optimization
- Adam, exponential + linear LR decay, gradient clipping (`max_norm=3.0`)

---

## Module Reference

| File | Role |
|---|---|
| `attributes.py` | 모든 하이퍼파라미터 단일 출처 |
| `graph_main.py` | 학습 루프, test cycle, collate_fn |
| `graph_model.py` | encode→process→decode, loss |
| `graph_networks.py` | MLP 블록, graph_net |
| `graph_builder.py` | edge local frame, boundary edge |
| `dataset.py` | 데이터 로딩, graph_data(), baking |
| `normalizer.py` | online z-score normalization |
| `preprocess_data.py` | DataPack을 .pt로 사전 저장 |

## Configuration

- `latent_size = 128`, `message_passing_steps = 10`
- `contact_distance = 0.75` mm (엣지 생성 반경)
- `lr = 1e-4`, `num_history = 5`
- Node types: `particle=0`, `hopper=1`, `roller1=2`, `roller2=3`
- 공간 도메인: X/Y ∈ [-50, 50] mm, Z ∈ [0, 100] mm (원본 m → ×1000)

---

## Issue & Fix History

### ✅ #1–#6 (2026-04-16) — 그래프 구조 버그 수정, re-baking 완료

| Issue | 내용 | 수정 |
|---|---|---|
| #1 | PM edge local frame = 0 | mesh face normal 기반 Gram-Schmidt frame 구성 |
| #2 | PM edge feature 0-padding | `~pairwise_mask`로 dx_local/dv_local 투영 |
| #3 | Normalizer baking 불일치 | `bake_mode=True`로 raw feature 저장, `__getitem__`에서 정규화 |
| #4 | collate_fn particle_indices offset | batch element별 node_offset 누적 |
| #5 | log/exp 수치 불안정 | `reverse_output()`에 `clamp(max=10.0)` 추가 |
| #6 | PP frame b ∥ a | velocity 기반 b'_ij + Gram-Schmidt + cross(b_perp, a) |

---

### ✅ #7–#10 (2026-04-19 ~ 04-24) — Y축 발산 조사

**증상:** 모든 실험에서 multi-step rollout Y축 발산. Loss 4~6 정체.

**조사 결과:**
- Loss 함수 변경 (plain MSE → per-axis → relative RMSE) 모두 효과 없음
- `target_normalizer` Y std = 3.151, X std = 0.039 → **80배 비대칭** 발견
- gravity 분리(Exp C) 시도 → mean만 변화, std는 동일 → 근본 원인 아님
- Y std 3.151의 진짜 원인 조사 → **Issue #11로 이어짐**

---

### ✅ #11 (2026-04-26) — 허수 target_acc 제거 ← 핵심 버그

**원인:**
도메인을 이탈하는 입자는 다음 timestep에서 `pos = (0,0,0)`으로 리셋됨.
중앙차분 공식 `acc = pos[t+1] - 2·pos[t] + pos[t-1]`에서:
```
acc_Y = 0 - 2·(-38.1) + (-35.9) = +40.3  ← 완전히 허수값
```
이 허수 acc가 `log(43) ≈ 3.76` → **normalizer Y std = 3.151의 직접 원인**.

**수정:** `graph_data()`에서 `next_particle_id > 0`인 입자만 `valid_particle_indices`로 선별, target/loss 계산에만 사용.

**효과:**
- Y std: 3.151 → 0.029 (**108배 감소**)
- Y/X std 비율: 80.8× → 3.9×
- Loss: 4~6 → 0.7~1.0
- 600k step에서 처음으로 초반 rollout 거동 안정화

---

### 🔄 현재 실험 — Exp D (2026-04-27 진행 중)

**배경:** Issue #11 fix 후에도 rollout 후반 발산. edge_decoder가 Y 방향 force를 충분히 생성 못함 (자유낙하 구간 PP/PM edge = 0).

**설계:** node_decoder를 처음부터 동시 학습 (Phase gating 제거) + node_penalty + momentum_loss로 edge_decoder 역할 보호 및 운동량 편향 억제.

```python
# 변경 전 (Phase gating)
if step < 100k:
    output = edge_output
else:
    output = edge_output + 0.1 * node_residual

# 변경 후 (Exp D - 2026-04-29)
output = edge_output + node_residual
loss_total = loss_main + 0.1 * node_penalty + 0.01 * momentum_loss
# momentum_loss: node_residual 전체 합산이 특정 방향으로 치우치지 않도록
```

**10k step 결과:**
```
node/edge ratio = 0.92  ✅ (균형 유지)
node_residual Y std = 0.344  (X=0.072, Z=0.099)
→ node_decoder가 Y 방향 외력(중력/낙하) 담당, edge_decoder는 접촉력(X,Z) 집중
```

**모니터링 지표:** 매 10k step DEBUG 로그
- `node/edge ratio < 1.0` → 정상
- `node/edge ratio 1.0~5.0` → 관찰
- `node/edge ratio > 5.0` → `NODE_PENALTY_WEIGHT` 0.1 → 0.3으로 상향
- `momentum_residual (X,Y,Z)`: 절댓값이 지속 증가하면 `MOMENTUM_LOSS_WEIGHT` 0.01 → 0.05로 상향

---

## Paper vs 현재 구현 차이 (Non-blocking)

| 항목 | 논문 | 현재 구현 |
|---|---|---|
| Spatiotemporal MP | 각 MP step에서 position/velocity Euler 적분 | latent space만 업데이트 |
| Boundary | ghost node reflection | mesh triangulation + closest point 투영 |
| Node decoding | inverse mass/inertia 분리 출력 | Edge + External Force Decoder 합산 |
