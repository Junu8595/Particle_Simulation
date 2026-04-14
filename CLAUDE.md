# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Graph Neural Network-based DEM (Discrete Element Method) particle simulator. Predicts particle dynamics in industrial equipment (hoppers, rollers) by learning particle interactions through graph message-passing networks with local coordinate frame invariance. Based on the scalarization-vectorization paradigm from the DYNAMI-CAL GRAPHNET paper (Nature Communications, 2026).

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

## Known Issues & Fix Plan

> Diagnosed via full code analysis + DYNAMI-CAL GRAPHNET paper comparison.
> Issues are ordered by priority. **#1 and #2 must be fixed together** (same code block in `dataset.py`).
> **#3 requires re-baking** all `.pt` files after fix.

### Current Symptoms

1. **Mesh(Roller) 관통**: Particle이 Roller/Hopper 표면을 뚫고 나감
2. **중력 미학습**: Particle이 깃털처럼 느리게 떨어짐
3. **비정상 튕김**: 간헐적으로 Particle이 예기치 못한 곳으로 튕겨져 나감

### Fix Priority

| Order | Issue | Severity | Files to Modify | Expected Impact |
|-------|-------|----------|-----------------|-----------------|
| 1 | #1 Contact edge local frame = 0 | CRITICAL | `dataset.py`, `graph_builder.py` | Roller 관통 해결 |
| 2 | #2 Contact edge feature 0-padding | CRITICAL | `dataset.py` | Mesh 방향/속도 학습 |
| 3 | #3 Normalizer baking 불일치 | CRITICAL | `preprocess_data.py`, `dataset.py` | 중력/튕김 해결 |
| 4 | #4 collate_fn particle_indices | MEDIUM | `graph_main.py` | batch>1 지원 |
| 5 | #5 Log/Exp 수치 불안정 | LOW | `dataset.py` | 안정성 향상 |

---

### Issue #1 [CRITICAL] Contact Edge Local Frame = (0,0,0)

**Where**: `dataset.py` → `graph_data()`, around line 361-390

**Bug**: `edge_a`, `edge_b`, `edge_c` are initialized as `torch.zeros()` and only filled for PP edges (indices `[:num_pairwise_edges]`). Contact (PM) edges remain all zeros.

**Effect**: In `graph_model.py` decoder, `f_ij = s1*edge_a + s2*edge_b + s3*edge_c` → **f_ij = 0** for all contact edges → mesh forces are completely ignored → particles pass through rollers.

**Fix**: Build a local frame for contact edges using mesh face normals from `build_boundary_edge()`.

**Implementation** — add this block in `dataset.py` `graph_data()` after the PP local frame section:
```python
# === Contact edge local frame (after the PP local frame block) ===
num_contact_edges = num_total_edges - num_pairwise_edges
if num_contact_edges > 0:
    contact_rel_pos = pos[senders[num_pairwise_edges:]] - pos[receivers[num_pairwise_edges:]]
    contact_a = graph_builder.safe_normalize(contact_rel_pos)

    # mesh_norm[:, :3] = face normal (returned by build_boundary_edge)
    contact_normal = mesh_norm[:, :3]

    # Gram-Schmidt: remove a-component from normal
    a_dot_n = torch.sum(contact_normal * contact_a, dim=1, keepdim=True)
    normal_perp = contact_normal - a_dot_n * contact_a
    contact_b = graph_builder.safe_normalize(normal_perp)

    # fallback for degenerate cases (normal parallel to a)
    b_degen = torch.norm(normal_perp, dim=1) < 1e-12
    if b_degen.any():
        contact_b[b_degen] = graph_builder.build_fallback_b_from_a(contact_a[b_degen])

    contact_c = graph_builder.safe_normalize(torch.cross(contact_a, contact_b, dim=-1))

    edge_a[num_pairwise_edges:] = contact_a
    edge_b[num_pairwise_edges:] = contact_b
    edge_c[num_pairwise_edges:] = contact_c
```

**Verify**: After fix, check that `edge_a[num_pairwise_edges:]` has no zero-rows and that `(a·b)`, `(a·c)`, `(b·c)` are all near zero (orthogonality).

---

### Issue #2 [CRITICAL] Contact Edge Feature 6/7 Dims = 0

**Where**: `dataset.py` → `graph_data()`, around line 399-414

**Bug**: `dx_local` and `dv_local` are computed only for `pairwise_mask == True`. Contact edges get `[dist, 0, 0, 0, 0, 0, 0]` as their 7D edge feature.

**Effect**: Edge encoder receives no directional or velocity information for PM edges. Network cannot learn mesh interaction direction.

**Fix**: Apply the same local-frame projection to contact edges. Add after the existing PP block:
```python
cm = ~pairwise_mask  # contact mask
if cm.any():
    dx_local[cm, 0] = torch.sum(rel_pos[cm] * edge_a[cm], dim=1)
    dx_local[cm, 1] = torch.sum(rel_pos[cm] * edge_b[cm], dim=1)
    dx_local[cm, 2] = torch.sum(rel_pos[cm] * edge_c[cm], dim=1)

    dv_local[cm, 0] = torch.sum(rel_vel[cm] * edge_a[cm], dim=1)
    dv_local[cm, 1] = torch.sum(rel_vel[cm] * edge_b[cm], dim=1)
    dv_local[cm, 2] = torch.sum(rel_vel[cm] * edge_c[cm], dim=1)
```

**Note**: This fix depends on Issue #1 being applied first (contact edges need valid `edge_a/b/c`).

---

### Issue #3 [CRITICAL] Pre-baking Normalizer ≠ Training Normalizer

**Where**:
- `preprocess_data.py` line 17-22: creates dummy normalizer with `max_accumulations=1`
- `dataset.py` `graph_data()`: calls `normalizer.forward(x, accumulate=True)` which normalizes + stores stats
- `dataset.py` `__getitem__()`: loads already-normalized `.pt` files
- `graph_main.py` line 204-206: creates fresh normalizer with `max_accumulations=1e6`

**Bug**: Baked `.pt` files contain features normalized by a dummy normalizer (only 1 sample of stats). Training creates a new normalizer that never sees raw features (only pre-normalized data). The two normalizers have completely different mean/std → `target_normalizer.inverse()` produces wrong scales at inference time.

**Effect**: Acceleration predictions are scaled incorrectly → gravity too weak ("feather-fall") or too strong ("explosion").

**Recommended Fix (Direction A)**: Save raw (un-normalized) features in `.pt` files. Apply normalization at training time in `__getitem__()`.

**Step 1** — `preprocess_data.py`: Don't apply normalization during baking.
Add a `bake_mode` flag to `gns_dataset` that skips normalizer calls in `graph_data()`:
```python
# In dataset.py, modify graph_data() normalizer calls:
if not getattr(self, 'bake_mode', False):
    edge_features = self.edge_normalizer(edge_features, accumulate=True)
    node_features = self.node_normalizer(node_features, accumulate=True)
    normalized_target = self.target_normalizer.forward(target_acc[:particle_indices.shape[0]], accumulate=True)
    normalized_target = self.target_normalizer.forward(target_acc, accumulate=False)
    # ... log transform ...
else:
    # In bake mode: skip normalization, store raw features
    normalized_target = target_acc  # raw acceleration, no normalization, no log transform
```
In `preprocess_data.py`, set `data_set.bake_mode = True` before the baking loop.

**Step 2** — `dataset.py` `__getitem__()`: Apply normalization when loading baked data:
```python
def __getitem__(self, idx):
    if self.mode == 'train':
        data_pack = torch.load(f'baked_training_data/step_{idx}.pt', weights_only=False)

        # Apply training normalizer to raw features
        node_features = self.node_normalizer(data_pack.nodepack.node_features, accumulate=True)
        edge_features = self.edge_normalizer(data_pack.edgepack.edge_features, accumulate=True)

        # Target: normalize then log-transform
        target_acc = data_pack.targetpack.target_acc
        particle_idx = data_pack.nodepack.particle_indices
        _ = self.target_normalizer.forward(target_acc[:particle_idx.shape[0]], accumulate=True)
        normalized_target = self.target_normalizer.forward(target_acc, accumulate=False)
        nt_sign = torch.where(normalized_target >= 0.0, 1, -1)
        normalized_target = torch.log(normalized_target.abs() + 1) * nt_sign

        # Reconstruct DataPack with normalized features
        new_np = data_pack.nodepack._replace(node_features=node_features)
        new_ep = data_pack.edgepack._replace(edge_features=edge_features)
        new_tp = data_pack.targetpack._replace(normalized_target=normalized_target)
        return DataPack(new_np, new_ep, new_tp)
```

**Step 3** — Re-bake all data: `python preprocess_data.py`

**Step 4** — `graph_main.py` pre-accumulation: The loop at line 334 calls `data_set[i]` which now triggers normalizer accumulation automatically. No change needed.

---

### Issue #4 [MEDIUM] collate_fn Missing particle_indices Offset

**Where**: `graph_main.py` → `gns_collate_fn()`, line 270

**Bug**: `new_nodepack = batch[0].nodepack._replace(node_features=node_features)` keeps `particle_indices` and `next_particle_indices` from the first batch element only. With `batch_size > 1`, loss computation references wrong nodes.

**Current impact**: `batch_size=1` → no effect now. Must fix before increasing batch size.

**Fix**: Accumulate and offset `particle_indices` / `next_particle_indices` in the existing loop:
```python
all_particle_indices, all_next_particle_indices = [], []
# ... inside the existing for d in batch loop:
    all_particle_indices.append(d.nodepack.particle_indices + node_offset)
    all_next_particle_indices.append(d.nodepack.next_particle_indices + node_offset)

particle_indices = torch.cat(all_particle_indices, dim=0)
next_particle_indices = torch.cat(all_next_particle_indices, dim=0)

new_nodepack = NodePack(node_features, particle_indices, next_particle_indices,
                        None, None, None)  # hopper/roller indices not used in training
```

---

### Issue #5 [LOW] Log/Exp Numerical Instability

**Where**: `dataset.py` line 347 (log transform) and `reverse_output()` line 534 (exp inverse)

**Bug**: `exp(|x|) - 1` amplifies small network errors exponentially. Combined with Issue #3, causes extreme values.

**Fix**: Add clamp in `reverse_output()`:
```python
output = torch.clamp(output.abs(), max=10.0)  # cap at e^10 ≈ 22026
output = (torch.pow(np.e * torch.ones_like(output), output) - 1) * output_sign
```

---

## Paper vs Current Code: Future Improvements (Non-blocking)

These differences from the DYNAMI-CAL GRAPHNET paper affect performance but are NOT the cause of current training failure:

1. **b'_ij construction**: Paper uses velocity + angular_velocity; current code uses reverse edge relative position only
2. **Spatiotemporal message passing**: Paper updates position/velocity via Euler integration at each MP step; current code only updates latent space
3. **Boundary handling**: Paper uses ghost node reflection (mesh-free); current code uses mesh triangulation + closest point projection
4. **Node-level decoding**: Paper decodes inverse mass/inertia from node embeddings; current code decodes acceleration directly from edge forces
