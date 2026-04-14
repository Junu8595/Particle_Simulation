import torch, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

pt_path = r'C:\Users\AISDL_PJW\Projects\Particle_Simulation\baked_training_data\step_0.pt'
data = torch.load(pt_path, weights_only=False)
ep  = data.edgepack
tp  = data.targetpack

contact_mask = ~ep.pairwise_mask
num_pp = ep.pairwise_mask.sum().item()
num_pm = contact_mask.sum().item()
print(f"PP edges: {num_pp}  PM edges: {num_pm}")

print("\n[1] contact edge_a/b/c zero-row check")
for name, vec in [('edge_a', ep.edge_a), ('edge_b', ep.edge_b), ('edge_c', ep.edge_c)]:
    norms = vec[contact_mask].norm(dim=1)
    zr = (norms < 1e-8).sum().item()
    print(f"  {name}: zero_rows={zr}/{num_pm}  norm min={norms.min():.6f} max={norms.max():.6f}")

print("\n[2] edge feature 7D (contact edges)")
ef = ep.edge_features
cev = ef[contact_mask]
print(f"  NaN={ef.isnan().sum().item()}  Inf={ef.isinf().sum().item()}")
labels = ['dist', 'dx_a', 'dx_b', 'dx_c', 'dv_a', 'dv_b', 'dv_c']
for d in range(7):
    col = cev[:, d]
    print(f"  dim[{d}] {labels[d]:5s}: min={col.min():.6f}  max={col.max():.6f}  all_zero={(col.abs()<1e-8).all().item()}")

print("\n[3] target fields")
ta = tp.target_acc
nt = tp.normalized_target
print(f"  target_acc == normalized_target (allclose): {torch.allclose(ta, nt)}")
print(f"  target_acc std={ta.std():.6f}  min={ta.min():.6f}  max={ta.max():.6f}")

print("\n[4] orthogonality |a.b| |a.c| |b.c| (contact edges)")
a = ep.edge_a[contact_mask]
b = ep.edge_b[contact_mask]
c = ep.edge_c[contact_mask]
ab = (a*b).sum(dim=1).abs()
ac = (a*c).sum(dim=1).abs()
bc = (b*c).sum(dim=1).abs()
print(f"  |a.b|: max={ab.max():.4e}  violate(>=1e-3)={(ab>=1e-3).sum().item()}/{num_pm}")
print(f"  |a.c|: max={ac.max():.4e}  violate(>=1e-6)={(ac>=1e-6).sum().item()}/{num_pm}")
print(f"  |b.c|: max={bc.max():.4e}  violate(>=1e-6)={(bc>=1e-6).sum().item()}/{num_pm}")

c1 = all((vec[contact_mask].norm(dim=1) >= 1e-8).all().item() for vec in [ep.edge_a, ep.edge_b, ep.edge_c])
c2 = not ef.isnan().any() and not ef.isinf().any() and all((cev[:,d].abs() > 1e-8).any().item() for d in [0,1,4,5,6])
c3 = torch.allclose(ta, nt)
c4 = (ab.max() < 1e-3)
PASS = lambda ok: "PASS" if ok else "FAIL"
print("\n===== VERDICT =====")
print(f"  [1] no zero-rows: {PASS(c1)}")
print(f"  [2] 7D valid: {PASS(c2)}")
print(f"  [3] raw stored: {PASS(c3)}")
print(f"  [4] orthogonality: {PASS(c4)}")
