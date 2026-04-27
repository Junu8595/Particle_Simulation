"""
Unit test: Issue #11 — valid_particle_indices 검증

두 가지 방식으로 검증:
  1. [raw]   raw .npy 데이터에서 직접 계산 (re-baking 전 실행 가능)
  2. [baked] re-baked step_100.pt에서 필드 유무 + 카운트 확인 (re-baking 후 실행)

기대 결과: step 100 기준으로 ~18개의 domain-exit 입자가 제외됨.
"""
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attributes

STEP_IDX = 100
BAKED_PATH = f'baked_training_data/step_{STEP_IDX}.pt'


def test_raw():
    """raw .npy에서 particle_id / next_particle_id 로드 후 valid_particle_indices 직접 계산."""
    print("=" * 60)
    print("[raw] 원시 데이터 기반 검증")
    print("=" * 60)

    device = 'cpu'
    _, training_attr = attributes.attribute(device)
    ds_props = training_attr[2]  # DataParameterPack

    # dataset 초기화 (normalizer 더미)
    import normalizer as norm_module
    from dataset import gns_dataset
    node_norm = norm_module.online_normalizer('node', 25, device='cpu')
    edge_norm = norm_module.online_normalizer('edge', 7, device='cpu')
    tgt_norm  = norm_module.online_normalizer('target', 3, device='cpu')
    ds = gns_dataset(ds_props, [node_norm, edge_norm, tgt_norm], device='cpu', mode='train')
    ds.load_dataset(ds_props.ds_path + ds_props.training_path)

    file_idx = ds.iterator[STEP_IDX, 0]
    iter_idx = ds.iterator[STEP_IDX, 1]
    print(f"  step {STEP_IDX} → file_idx={file_idx}, iter_idx={iter_idx}")

    particle_id_data = ds.dataset[file_idx][4]          # (T, N_max)
    particle_pos_data = ds.dataset[file_idx][2]         # (T, N_max, 3)

    particle_id_cur  = particle_id_data[iter_idx].float()
    particle_id_next = particle_id_data[iter_idx + 1].float()

    particle_indices       = particle_id_cur.nonzero(as_tuple=False).reshape(-1)
    next_particle_indices  = particle_id_next.nonzero(as_tuple=False).reshape(-1)
    valid_particle_indices = particle_indices[particle_id_next[particle_indices] > 0]

    n_cur    = len(particle_indices)
    n_next   = len(next_particle_indices)
    n_valid  = len(valid_particle_indices)
    n_exit   = n_cur - n_valid

    print(f"  particle_indices (t)         : {n_cur}")
    print(f"  next_particle_indices (t+1)  : {n_next}")
    print(f"  valid_particle_indices       : {n_valid}")
    print(f"  domain-exit (excluded)       : {n_exit}")

    # 이탈 입자의 Y acc가 실제로 비정상적으로 큰지 확인
    exiting = particle_indices[particle_id_next[particle_indices] == 0]
    if len(exiting) > 0:
        prev_pos = particle_pos_data[max(iter_idx - 1, 0)]
        cur_pos  = particle_pos_data[iter_idx]
        nxt_pos  = particle_pos_data[iter_idx + 1]

        acc_exit  = nxt_pos[exiting] - 2 * cur_pos[exiting] + prev_pos[exiting]
        acc_valid = nxt_pos[valid_particle_indices] - 2 * cur_pos[valid_particle_indices] + prev_pos[valid_particle_indices]

        print(f"\n  [spurious acc 검증]")
        print(f"  이탈 입자 |Y-acc| mean={acc_exit[:,1].abs().mean():.4f}  max={acc_exit[:,1].abs().max():.4f}")
        print(f"  유효 입자 |Y-acc| mean={acc_valid[:,1].abs().mean():.6f}  max={acc_valid[:,1].abs().max():.4f}")
        ratio = acc_exit[:,1].abs().mean() / (acc_valid[:,1].abs().mean() + 1e-8)
        print(f"  비율 (이탈/유효): {ratio:.1f}x")
        assert ratio > 10, f"이탈 입자 acc가 유효 입자 대비 10배 이상이어야 함 (실제: {ratio:.1f}x)"

    assert n_exit >= 0, "valid count가 particle count를 초과할 수 없음"
    print(f"\n  [raw] ✅ 통과: step {STEP_IDX}에서 domain-exit {n_exit}개 제외 확인")
    return n_exit


def test_baked():
    """re-baked .pt 파일에서 valid_particle_indices 필드 유무 및 카운트 검증."""
    print("\n" + "=" * 60)
    print("[baked] re-baked .pt 파일 기반 검증")
    print("=" * 60)

    if not os.path.exists(BAKED_PATH):
        print(f"  ⚠️  {BAKED_PATH} 없음 — re-baking 후 다시 실행하세요")
        return None

    data_pack = torch.load(BAKED_PATH, weights_only=False)
    np_ = data_pack.nodepack

    assert 'valid_particle_indices' in np_._fields, \
        "NodePack에 valid_particle_indices 필드 없음 — re-baking 필요"

    n_particle = len(np_.particle_indices)
    n_valid    = len(np_.valid_particle_indices)
    n_excluded = n_particle - n_valid

    print(f"  particle_indices count       : {n_particle}")
    print(f"  valid_particle_indices count : {n_valid}")
    print(f"  excluded (domain-exit)       : {n_excluded}")

    assert n_excluded >= 0, "valid count가 particle count를 초과할 수 없음"

    # valid_particle_indices가 particle_indices의 부분집합인지 확인
    pi_set = set(np_.particle_indices.tolist())
    vpi_set = set(np_.valid_particle_indices.tolist())
    assert vpi_set.issubset(pi_set), "valid_particle_indices가 particle_indices의 부분집합이어야 함"

    print(f"\n  [baked] ✅ 통과: step {STEP_IDX}에서 domain-exit {n_excluded}개 제외 확인")
    return n_excluded


if __name__ == '__main__':
    n_exit_raw = test_raw()
    n_exit_baked = test_baked()

    if n_exit_baked is not None:
        assert n_exit_raw == n_exit_baked, \
            f"raw({n_exit_raw})와 baked({n_exit_baked}) 제외 수가 일치해야 함"
        print(f"\n✅ raw/baked 일치: {n_exit_raw}개 제외")
    else:
        print(f"\nre-baking 전 raw 검증만 완료: {n_exit_raw}개 제외")
