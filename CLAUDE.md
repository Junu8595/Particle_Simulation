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
   - **Encoder**: 25D→128D nodes, 7D→128D edges (MLPs with LayerNorm)
   - **Processor**: 10 message-passing steps with attention + residual connections
   - **Decoder**: Two separate edge MLPs — `edge_decoder_pp` for PP, `edge_decoder_pm` for PM
   - Output: 3 scalar coefficients `(s1, s2, s3)` per edge → force vector `f_ij = s1·a + s2·b + s3·c`

5. **Loss**: Edge forces aggregated to nodes via `scatter_add`; MSE on normalized acceleration for particle nodes only. Log-transformed targets with relative RMSE loss.

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
| `graph_model.py` | Graph wrapper: encode→process→decode; dual decoder (PP/PM); loss computation |
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

## ✅ All Issues Resolved (as of 2026-04-16)

Issues #1–#7 fully applied. Re-baked data (6176 `.pt` files) uploaded to remote server.
Training ready to run: `cd /home/ssdl/PJW/Particle_Simulation && git pull && python graph_main.py`

| Issue | Commit | Summary |
|-------|--------|---------|
| #1 Contact edge local frame = 0 | `417de37` | Gram-Schmidt frame (a,b,c) built for PM edges via mesh face normal |
| #2 Contact edge feature 0-padding | `417de37` | dx_local/dv_local projected for contact edges via `cm = ~pairwise_mask` |
| #3 Normalizer baking mismatch | `417de37` | `bake_mode=True` stores raw features; normalization applied in `__getitem__()` |
| #4 collate_fn particle_indices offset | `80e3bc8` | particle_indices / next_particle_indices accumulated with node_offset per batch element |
| #5 Log/Exp numerical instability | `80e3bc8` | clamp(max=10.0) added in `reverse_output()` before exp |
| #6 PP local frame b ∦ a | `192bed3` | velocity-based b'_ij (DYNAMI-CAL 방식) + Gram-Schmidt + cross(b_perp, a) |
| #7 Y-axis bias in loss | `759be33` | relative RMSE → plain MSE; normalizer의 per-axis z-score 정규화로 충분 |

**Validation on real step_0 data (2026-04-15):**
- PP edges: 144,161 / PM edges: 15,982
- PP `max |a·b| = 0.000000` ✅ (이전: 1.000000)
- PM `max |a·b| = 0.000250` ✅
- All frame vectors unit-norm, b_degenerate = 0

---

## preprocess_data.py — GPU 가속 (as of 2026-04-15, commit `b4dc180`)

`preprocess_data.py`는 GPU를 활용하도록 최적화됨. `self.device`가 `cuda:0`이면 자동으로 GPU 연산.

**변경 내용:**
- `build_reverse_edge_index()`: Python dict 루프(144K×2) → `searchsorted` 벡터화 O(E log E)
- `build_edge_local_frame_3d()` antisymmetry 루프: Python for → 텐서 인덱싱
- `radius_graph`: `sub_particle_pos.to(self.device)` 후 `.cpu()` 복귀
- `build_boundary_edge`: `'cpu'` 하드코딩 → `self.device` 전달 후 출력 `.cpu()` 복귀

**벤치마크 (RTX 4070 Super):** ~4s/step(CPU) → 0.41s/step(GPU), **10x 향상**, 6176 steps ≈ 40분

**Re-bake 명령:**
```bash
python preprocess_data.py   # attributes.py의 device 설정에 따라 자동으로 GPU/CPU 선택
```

---

## 🟡 Active Issues

없음. 현재 알려진 버그 없음.

---

## Paper vs Current Code: Future Improvements (Non-blocking)

These differences from the DYNAMI-CAL GRAPHNET paper affect performance but do NOT block training:

1. ~~**b'_ij construction**: Paper uses velocity + angular_velocity~~ → **#6에서 velocity 기반으로 수정 완료**
2. **Spatiotemporal message passing**: Paper updates position/velocity via Euler integration at each MP step; current code only updates latent space
3. **Boundary handling**: Paper uses ghost node reflection (mesh-free); current code uses mesh triangulation + closest point projection
4. **Node-level decoding**: Paper decodes inverse mass/inertia from node embeddings; current code decodes acceleration directly from edge forces
