import torch
import torch_geometric
import torch_scatter

def build_node_type(node_type):
    types = torch.tensor([0,1,2,3])

    one_hot_vectors = torch.zeros((node_type.shape[0], types.shape[0])).to(node_type.device)

    for i,t in enumerate(types):
        one_hot_vectors[:,i] = torch.where(node_type == t, 1, 0)

    return one_hot_vectors

def build_geo_edge(receivers, senders, distance, indices):

    if receivers.shape[0] == 0:
        return receivers, senders, distance

    mask = torch.isin(senders, indices)
    idx = mask.nonzero().squeeze()
    r = receivers[idx].long()
    d = distance[idx]

    dim_size = int(receivers.max().item()) + 1

    if r.ndim == 0:
        idx = idx.unsqueeze(0)
        r= r.unsqueeze(0)
        d = d.unsqueeze(0)


    min_vals, argmins = torch_scatter.scatter_min(d, r, dim=0, dim_size=dim_size)
    valid = argmins != -1

    K = idx.numel()
    pos = (argmins >= 0) & (argmins < K)
    pos = argmins[valid & pos]

    mask_idx = idx[pos]

    mask_d = torch.zeros_like(mask)
    mask_d[mask_idx] = True

    final_mask = (~mask) | mask_d
    return receivers[final_mask], senders[final_mask], distance[final_mask]


def make_rotation_mat(tensor, rot_angle, direction):
    if direction == 'x':
        rot_mat = torch.tensor([[1,0,0],
                                [0,torch.cos(rot_angle),-torch.sin(rot_angle)],
                                [0,torch.sin(rot_angle),torch.cos(rot_angle)]]).float()
        
    elif direction == 'y':
        rot_mat = torch.tensor([[torch.cos(rot_angle),0,torch.sin(rot_angle)],
                                [0,1,0],
                                [-torch.sin(rot_angle),0,torch.cos(rot_angle)]]).float()
        
    else:
        rot_mat = torch.tensor([[torch.cos(rot_angle), -torch.sin(rot_angle),0],
                                [torch.sin(rot_angle),torch.cos(rot_angle),0],
                                [0,0,1]]).float()
        
    rot_mat = rot_mat.expand((tensor.shape[0],-1,-1))

    return rot_mat

def rotate_pos(tensor, rot_mat):
    rotated_tensor = torch.bmm(rot_mat, tensor[:,:,None]).squeeze(dim=-1)
    return rotated_tensor.reshape((rotated_tensor.shape[0],rotated_tensor.shape[1]))

def build_edge_features(pos_list, receivers, senders):
    features = []

    for pos in pos_list:
        rel_pos = pos[senders] - pos[receivers]
        rel_pos_norm = torch.norm(rel_pos, dim=1)[:,None]

        features.append(rel_pos)
        features.append(rel_pos_norm)

    return torch.hstack(features)
def safe_normalize(vec, eps=1e-12):
    """
    마지막 차원 기준으로 벡터 정규화
    """
    norm = torch.norm(vec, dim=-1, keepdim=True)
    return vec / torch.clamp(norm, min=eps)


def project_vector(vec, basis):
    return torch.sum(vec * basis, dim=-1, keepdim=True) * basis


def build_reverse_edge_index(receivers, senders):
    device = receivers.device
    E = receivers.shape[0]
    if E == 0:
        return torch.full((0,), -1, dtype=torch.long, device=device)

    N = int(max(receivers.max().item(), senders.max().item())) + 1

    # 각 edge (r, s)를 unique scalar ID로 인코딩
    fwd_ids = receivers * N + senders   # (E,)
    rev_ids = senders * N + receivers   # (E,) — 각 edge의 역방향 ID

    sorted_ids, sort_perm = torch.sort(fwd_ids)

    # 각 rev_id가 sorted_ids에서 어느 위치인지 탐색 (O(E log E))
    ins = torch.searchsorted(sorted_ids, rev_ids)
    ins_clamped = ins.clamp(0, E - 1)

    in_bounds = ins < E
    matched = sorted_ids[ins_clamped] == rev_ids
    valid = in_bounds & matched

    reverse_edge_idx = torch.full((E,), -1, dtype=torch.long, device=device)
    reverse_edge_idx[valid] = sort_perm[ins_clamped[valid]]

    return reverse_edge_idx


def build_a_ij(pos, receivers, senders, eps=1e-12):
    rel_pos = pos[senders] - pos[receivers]
    a = safe_normalize(rel_pos, eps)
    return a, rel_pos


def build_bprime_ij(pos,
                    vel,
                    receivers,
                    senders,
                    omega=None,
                    eps=1e-12):
    """
    term1 = normalize(v_j + v_i)
    term2 = normalize((v_j - v_i) x (r_j - r_i))
    optional:
        term3 = normalize(w_j + w_i)
        term4 = normalize((w_j - w_i) x (r_j - r_i))

    b'_ij = term1 + term2 (+ term3 + term4)

    입력:
    - pos: (N, 3)
    - vel: (N, 3)        
    - receivers: (E,)
    - senders: (E,)
    - omega: (N, 3) or None

    출력:
    - b_prime: (E, 3)
    """
    vi = vel[receivers]
    vj = vel[senders]

    ri = pos[receivers]
    rj = pos[senders]
    rij = rj - ri

    term1 = safe_normalize(vj + vi, eps)

    cross_v = torch.cross(vj - vi, rij, dim=-1)
    term2 = safe_normalize(cross_v, eps)

    b_prime = term1 + term2

    if omega is not None:
        wi = omega[receivers]
        wj = omega[senders]

        term3 = safe_normalize(wj + wi, eps)

        cross_w = torch.cross(wj - wi, rij, dim=-1)
        term4 = safe_normalize(cross_w, eps)

        b_prime = b_prime + term3 + term4

    return b_prime


def build_fallback_b_from_a(a, eps=1e-12):
    device = a.device
    dtype = a.dtype
    E = a.shape[0]

    ex = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand(E, -1)
    ey = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).expand(E, -1)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand(E, -1)

    dots_x = torch.abs(torch.sum(a * ex, dim=-1))
    dots_y = torch.abs(torch.sum(a * ey, dim=-1))
    dots_z = torch.abs(torch.sum(a * ez, dim=-1))

    basis_stack = torch.stack([dots_x, dots_y, dots_z], dim=1)
    min_axis = torch.argmin(basis_stack, dim=1)

    ref = ex.clone()
    ref[min_axis == 1] = ey[min_axis == 1]
    ref[min_axis == 2] = ez[min_axis == 2]

    ref_perp = ref - project_vector(ref, a)
    fallback_b = torch.cross(ref_perp, a, dim=-1)
    fallback_b = safe_normalize(fallback_b, eps)

    return fallback_b


def build_b_ij(a, b_prime, eps=1e-12):
    b_parallel = project_vector(b_prime, a)
    b_perp = b_prime - b_parallel

    cross_val = torch.cross(b_perp, a, dim=-1)
    cross_norm = torch.norm(cross_val, dim=-1)

    degenerate_mask = cross_norm < eps

    b = safe_normalize(cross_val, eps)

    if degenerate_mask.any():
        fallback_b = build_fallback_b_from_a(a[degenerate_mask], eps)
        b[degenerate_mask] = fallback_b

    return b, b_parallel, b_perp, degenerate_mask


def build_c_ij(a, b, b_parallel, eps=1e-12):
    c_raw = torch.cross(b_parallel, b, dim=-1)
    c_norm = torch.norm(c_raw, dim=-1)

    c_degenerate_mask = c_norm < eps

    c = safe_normalize(c_raw, eps)

    if c_degenerate_mask.any():
        c_fallback = torch.cross(a[c_degenerate_mask], b[c_degenerate_mask], dim=-1)
        c_fallback = safe_normalize(c_fallback, eps)
        c[c_degenerate_mask] = c_fallback

    return c, c_degenerate_mask


def project_vectors_to_edge_frame(vectors, a, b, c):
    """
    벡터들을 edge local frame (a, b, c) 위의 scalar 성분으로 projection
    이후 invariant / scalarized edge feature를 만들 때 사용

    입력:
    - vectors: (E, 3)
    - a, b, c: (E, 3)

    출력:
    - projected scalars: (E, 3)
      [:,0] = <v, a>
      [:,1] = <v, b>
      [:,2] = <v, c>
    """
    va = torch.sum(vectors * a, dim=-1, keepdim=True)
    vb = torch.sum(vectors * b, dim=-1, keepdim=True)
    vc = torch.sum(vectors * c, dim=-1, keepdim=True)

    return torch.hstack([va, vb, vc])

def build_edge_local_frame_3d(pos,
                              vel,
                              receivers,
                              senders,
                              eps=1e-12):
    """
    particle-particle directed edge마다 3D antisymmetric local frame (a_ij, b_ij, c_ij) 계산

    입력:
    - pos: (N, 3)
    - vel: (N, 3)  — 현재 스텝 속도 (build_bprime_ij에 사용)
    - receivers: (E,)
    - senders: (E,)

    출력:
    - edge_a: (E, 3)
    - edge_b: (E, 3)
    - edge_c: (E, 3)
    - reverse_edge_idx: (E,)
    - b_degenerate_mask: (E,)
    - c_degenerate_mask: (E,)
    """

    device = pos.device
    dtype = pos.dtype
    num_edges = receivers.shape[0]

    if num_edges == 0:
        empty_vec = torch.empty((0, 3), device=device, dtype=dtype)
        empty_idx = torch.empty((0,), device=device, dtype=torch.long)
        empty_mask = torch.empty((0,), device=device, dtype=torch.bool)
        return (
            empty_vec,
            empty_vec,
            empty_vec,
            empty_idx,
            empty_mask,
            empty_mask,
        )

    # reverse edge index
    reverse_edge_idx = build_reverse_edge_index(receivers, senders)

    # a_ij = normalized relative position
    edge_a, rel_pos = build_a_ij(pos, receivers, senders, eps)

    # -----------------------------
    # b_ij 구성 (velocity-based, DYNAMI-CAL GRAPHNET 방식)
    # b'_ij = normalize(v_j + v_i) + normalize((v_j - v_i) x r_ij)
    # 이후 Gram-Schmidt + cross(b_perp, a) 로 a와 직교하는 b를 얻음
    # -----------------------------
    b_prime = build_bprime_ij(pos, vel, receivers, senders, omega=None, eps=eps)

    edge_b, b_parallel, b_perp, b_degenerate_mask = build_b_ij(edge_a, b_prime, eps)

    # c_ij = cross(b_parallel, b) (build_c_ij 방식)
    edge_c, c_degenerate_mask = build_c_ij(edge_a, edge_b, b_parallel, eps)

    # antisymmetry 강제: i < j인 쌍에서 edge_a/b/c[j] = -edge_a/b/c[i] 복사 (벡터화)
    i_idx = torch.arange(num_edges, device=device)
    valid_rev = reverse_edge_idx >= 0
    j_idx = reverse_edge_idx.clamp(min=0)  # 인덱싱 안전을 위해 clamp

    primary_mask = valid_rev & (i_idx < j_idx)
    j_targets = j_idx[primary_mask]

    edge_a[j_targets] = -edge_a[primary_mask]
    edge_b[j_targets] = -edge_b[primary_mask]
    edge_c[j_targets] = -edge_c[primary_mask]
    b_degenerate_mask[j_targets] = b_degenerate_mask[primary_mask]
    c_degenerate_mask[j_targets] = c_degenerate_mask[primary_mask]

    return (
        edge_a,
        edge_b,
        edge_c,
        reverse_edge_idx,
        b_degenerate_mask,
        c_degenerate_mask,
    )

def build_boundary_edge(cells, particle_pos, mesh_pos, particle_indices, mesh_node_type, mesh_vel, device, pm_contact_distance):

    num_interaction_hopper = 5
    num_interaction_roller1 = 5
    num_interaction_roller2 = 5
    
    with torch.no_grad():
        # Ensure all input tensors are on the specified device
        cells = cells.to(device)
        particle_pos = particle_pos.to(device)
        mesh_pos = mesh_pos.to(device)
        particle_indices = particle_indices.to(device)
        mesh_node_type = mesh_node_type.to(device)
        mesh_vel = mesh_vel.to(device)

        diff = (torch.reshape(particle_pos,(-1,1,3)) - torch.reshape(mesh_pos[cells[:,0]], (1,-1,3)))
        
        edge0 = mesh_pos[cells[:,1]] - mesh_pos[cells[:,0]]
        edge1 = mesh_pos[cells[:,2]] - mesh_pos[cells[:,0]]

        normal_vector = torch.cross(edge0, edge1, dim=-1)
        normal_vector = normal_vector / ((normal_vector.norm(dim=-1, keepdim=True)) + 1e-12)

        a00 = torch.reshape(torch.sum(edge0 * edge0, 1), (1,-1))
        a01 = torch.reshape(torch.sum(edge0 * edge1, 1), (1,-1))
        a11 = torch.reshape(torch.sum(edge1 * edge1, 1), (1,-1))

        b0 = -torch.sum(diff*torch.reshape(edge0,(1,-1,3)),2)
        b1 = -torch.sum(diff*torch.reshape(edge1,(1,-1,3)),2)

        det = a00*a11 - a01*a01
        t0 = a01*b1 - a11*b0
        t1 = a01*b0 - a00*b1

        cond_1111=a11+b1-a01-b0>=a00-2.0*a01+a11
        t0_1111=torch.where(cond_1111, torch.as_tensor(1.0, device=device).float(), (a11+b1-a01-b0)/(a00-2.0*a01+a11))
        t1_1111=torch.where(cond_1111, torch.as_tensor(0.0, device=device).float(), 1.0-((a11+b1-a01-b0)/(a00-2.0*a01+a11)))

        t0_1110=torch.as_tensor(0.0, device=device).float()
        t1_1110=torch.as_tensor(1.0, device=device).float()

        cond_111=a11+b1-a01-b0<=0.0
        t0_111=torch.where(cond_111, t0_1110, t0_1111)
        t1_111=torch.where(cond_111, t1_1110, t1_1111)

        del cond_1111, cond_111, t0_1111, t1_1111, t0_1110, t1_1110

        cond_11011=b0>=0.0
        t0_11011=torch.where(cond_11011, torch.as_tensor(0.0, device=device).float(), -b0/a00)
        t1_11011=torch.as_tensor(0.0, device=device).float()

        t0_11010=torch.as_tensor(1.0, device=device).float()
        t1_11010=torch.as_tensor(0.0, device=device).float()

        cond_1101=a00+b0<=0.0
        t0_1101=torch.where(cond_1101, t0_11010, t0_11011)
        t1_1101=torch.where(cond_1101, t1_11010, t1_11011)

        del cond_11011, cond_1101, t0_11011, t1_11011, t0_11010, t1_11010

        #cond_1100=a11+b1-a01-b0>=a00-2.0*a01+a11
        cond_1100=((a00+b0)-(a01+b1))>=(a00-(2.0)*a01+a11)
        t0_1100=torch.where(cond_1100, torch.as_tensor(0.0, device=device).float(), 1.0-(((a00+b0)-(a01+b1))/(a00-(2.0)*a01+a11)))
        t1_1100=torch.where(cond_1100, torch.as_tensor(1.0, device=device).float(), ((a00+b0)-(a01+b1))/(a00-(2.0)*a01+a11))

        cond_110=a00+b0>a01+b1
        t0_110=torch.where(cond_110, t0_1100, t0_1101)
        t1_110=torch.where(cond_110, t1_1100, t1_1101)

        cond_11=t1<0.0
        t0_11=torch.where(cond_11, t0_110, t0_111)
        t1_11=torch.where(cond_11, t1_110, t1_111)

        del cond_1100, cond_110, cond_11, t0_1100, t1_1100, t0_1101, t1_1101, t0_110, t1_110, t0_111, t1_111

        cond_1011=b1>=0.0
        t0_1011=torch.as_tensor(0.0, device=device).float()
        t1_1011=torch.where(cond_1011, torch.as_tensor(0.0, device=device).float(), -b1/a11)

        t0_1010=torch.as_tensor(0.0, device=device).float()
        t1_1010=torch.as_tensor(1.0, device=device).float()

        cond_101=(a11+b1)<=0.0
        t0_101=torch.where(cond_101, t0_1010, t0_1011)
        t1_101=torch.where(cond_101, t1_1010, t1_1011)

        del cond_1011, cond_101, t0_1011, t1_1011, t0_1010, t1_1010

        cond_100=((a11+b1)-(a01+b0)) >= (a00-(2.0)*a01+a11)
        t0_100=torch.where(cond_100, torch.as_tensor(1.0, device=device).float(), ((a11+b1)-(a01+b0))/(a00-(2.0)*a01+a11))
        t1_100=torch.where(cond_100, torch.as_tensor(0.0, device=device).float(), 1.0-(((a11+b1)-(a01+b0))/(a00-(2.0)*a01+a11)))

        cond_10=(a11+b1)>(a01+b0)
        t0_10=torch.where(cond_10, t0_100, t0_101)
        t1_10=torch.where(cond_10, t1_100, t1_101)

        del cond_100, cond_10, t0_100, t1_100, t0_101, t1_101

        cond_1=t0<0.0
        t0_1=torch.where(cond_1, t0_10, t0_11)
        t1_1=torch.where(cond_1, t1_10, t1_11)

        t0_011=(a01*b1-a11*b0)*(1.0/det)
        t1_011=(a01*b0-a00*b1)*(1.0/det)

        del cond_1, t0_10, t1_10, t0_11, t1_11

        cond_0101=-b0>=a00
        t0_0101=torch.where(cond_0101, torch.as_tensor(1.0, device=device).float(), -b0/a00)
        t1_0101=torch.as_tensor(0.0, device=device).float()

        t0_0100=torch.as_tensor(0.0, device=device).float()
        t1_0100=torch.as_tensor(0.0, device=device).float()

        cond_010=b0 >= 0.0
        t0_010=torch.where(cond_010, t0_0100, t0_0101)
        t1_010=torch.where(cond_010, t1_0100, t1_0101)

        del cond_0101, cond_010, t0_0101, t1_0101, t0_0100, t1_0100

        cond_01=t1<0.0
        t0_01=torch.where(cond_01, t0_010, t0_011)
        t1_01=torch.where(cond_01, t1_010, t1_011)

        del cond_01, t0_010, t1_010, t0_011, t1_011

        cond_0011=-b1>=a11
        t0_0011=torch.as_tensor(0.0, device=device).float()
        t1_0011=torch.where(cond_0011, torch.as_tensor(1.0, device=device).float(), -b1/a11)

        t0_0010=torch.as_tensor(0.0, device=device).float()
        t1_0010=torch.as_tensor(0.0, device=device).float()

        cond_001=b1>=0.0
        t0_001=torch.where(cond_001, t0_0010, t0_0011)
        t1_001=torch.where(cond_001, t1_0010, t1_0011)

        del cond_0011, cond_001, t0_0011, t1_0011, t0_0010, t1_0010

        cond_00011=-b1>=a11
        t0_00011=torch.as_tensor(0.0, device=device).float()
        t1_00011=torch.where(cond_00011, torch.as_tensor(1.0, device=device).float(), -b1/a11)

        t0_00010=torch.as_tensor(0.0, device=device).float()
        t1_00010=torch.as_tensor(0.0, device=device).float()

        cond_0001=b1>=0.0
        t0_0001=torch.where(cond_0001, t0_00010, t0_00011)
        t1_0001=torch.where(cond_0001, t1_00010, t1_00011)

        del cond_00011, cond_0001, t0_00010, t1_00010, t0_00011, t1_00011

        cond_0000=-b0>=a00
        t0_0000=torch.where(cond_0000, torch.as_tensor(1.0, device=device).float(), -b0/a00)
        t1_0000=torch.as_tensor(0.0, device=device).float()

        cond_000=b0<0.0
        t0_000=torch.where(cond_000, t0_0000, t0_0001)
        t1_000=torch.where(cond_000, t1_0000, t1_0001)

        del cond_0000, cond_000, t0_0000, t1_0000, t0_0001, t1_0001

        cond_00=t1<0.0
        t0_00=torch.where(cond_00, t0_000, t0_001)
        t1_00=torch.where(cond_00, t1_000, t1_001)

        cond_0=t0<0.0
        t0_0=torch.where(cond_0, t0_00, t0_01)
        t1_0=torch.where(cond_0, t1_00, t1_01)

        cond=t0+t1<=det
        t0=torch.where(cond, t0_0, t0_1)
        t1=torch.where(cond, t1_0, t1_1)

        del cond_00, cond_0, cond, t0_000, t1_000, t0_001, t1_001, t0_00, t1_00, t0_01, t1_01, t0_0, t1_0, t0_1, t1_1

        vecShape=(-1,)+tuple(edge0.shape)
        factorShape=tuple(t0.shape)+(-1,)
        startPoint=torch.reshape(mesh_pos[cells[:,0]], vecShape)
        t0=torch.reshape(t0, factorShape)
        t1=torch.reshape(t1, factorShape)
        edge0=torch.reshape(edge0, vecShape)
        edge1=torch.reshape(edge1, vecShape)

        del a00, a01, a11, b0, b1, det

        minPoint=startPoint+t0*edge0+t1*edge1

        diffVec=torch.reshape(particle_pos, (-1,1,3))-minPoint
        dists=torch.sqrt(torch.sum(diffVec**2, 2))

        hopper_mask = torch.where(mesh_node_type[cells[:,0]] == 1, 1, 0).nonzero().squeeze()
        roll1_mask = torch.where(mesh_node_type[cells[:,0]] == 2, 1, 0).nonzero().squeeze()
        roll2_mask = torch.where(mesh_node_type[cells[:,0]] == 3, 1, 0).nonzero().squeeze()

        num_interaction_hopper = min(num_interaction_hopper, dists[:, hopper_mask].size(1))
        num_interaction_roller1 = min(num_interaction_roller1, dists[:, roll1_mask].size(1))
        num_interaction_roller2 = min(num_interaction_roller2, dists[:, roll2_mask].size(1))

        dmink_hopper, hop_indices = torch.topk(dists[:, hopper_mask], k = num_interaction_hopper, dim=1, largest=False, sorted=False) # (num_hop, num_interaction)
        dmink_roll1, r1_indices = torch.topk(dists[:, roll1_mask], k = num_interaction_roller1, dim=1, largest=False, sorted=False) # (num_roll1, num_interaction)
        dmink_roll2, r2_indices = torch.topk(dists[:, roll2_mask], k = num_interaction_roller2, dim=1, largest=False, sorted=False) # (num_roll2, num_interaction)

        hop_particle = torch.where(dmink_hopper <= torch.tensor(pm_contact_distance, device=device)) # hop_particle[0] : dim=0의 idx [1] : dim=1의 idx
        roll1_particle = torch.where(dmink_roll1 <= torch.tensor(pm_contact_distance, device=device))
        roll2_particle = torch.where(dmink_roll2 <= torch.tensor(pm_contact_distance, device=device))

        hop_mask = torch.isin(hop_particle[0], particle_indices)
        hopper_receiver = hop_particle[0][hop_mask]
        hopper_sender = hopper_mask[hop_indices[hop_particle[0][hop_mask], hop_particle[1][hop_mask]]]
        hopper_pos = minPoint[hopper_receiver, hopper_sender]

        r1_mask = torch.isin(roll1_particle[0], particle_indices)
        roll1_receiver = roll1_particle[0][r1_mask]
        roll1_sender = roll1_mask[r1_indices[roll1_particle[0][r1_mask],roll1_particle[1][r1_mask]]]
        roll1_pos = minPoint[roll1_receiver, roll1_sender]

        r2_mask = torch.isin(roll2_particle[0], particle_indices)
        roll2_receiver = roll2_particle[0][r2_mask]
        roll2_sender = roll2_mask[r2_indices[roll2_particle[0][r2_mask],roll2_particle[1][r2_mask]]]
        roll2_pos = minPoint[roll2_receiver, roll2_sender]

        normal = torch.vstack([normal_vector[hopper_sender], normal_vector[roll1_sender], normal_vector[roll2_sender]])
        normal_sign = torch.sign(normal)
        f_on = (normal_sign[:, 0] + 1) + (normal_sign[:, 1] + 1) * 3 + (normal_sign[:, 2] + 1) * 9

        m_normal = - normal
        m_normal_sign = torch.sign(m_normal)
        f_om = (m_normal_sign[:, 0] + 1) + (m_normal_sign[:, 1] + 1) * 3 + (m_normal_sign[:, 2] + 1) * 9

        normal_mask = (f_on <= f_om)
        normal_mask = normal_mask.unsqueeze(-1)

        first_norm = torch.where(normal_mask, normal, m_normal)
        second_norm = torch.where(normal_mask, m_normal, normal)

    norm = torch.cat([first_norm, second_norm], dim= -1)


    # dmin_hopper, hop_indices = torch.min(dists[:,hopper_mask], dim=1)
    # hop_min_indices = hopper_mask[hop_indices]
    # dmin_roll1, r1_indices = torch.min(dists[:,roll1_mask], dim=1)
    # r1_min_indices = roll1_mask[r1_indices]
    # dmin_roll2, r2_indices = torch.min(dists[:,roll2_mask], dim=1)
    # r2_min_indices = roll2_mask[r2_indices]

    # #takeDist=torch.where(dists<neighborCutoff)
    # hop_particle = torch.where(dmin_hopper<=torch.tensor(contact_distance, device=device))[0]
    # roll1_particle = torch.where(dmin_roll1<=torch.tensor(contact_distance, device=device))[0]
    # roll2_particle = torch.where(dmin_roll2<=torch.tensor(contact_distance, device=device))[0]

    # hop_mask = torch.isin(hop_particle, particle_indices)
    # hopper_receiver = hop_particle[hop_mask]
    # hopper_pos = minPoint[hopper_receiver,hop_min_indices[hopper_receiver],:]

    # r1_mask = torch.isin(roll1_particle, particle_indices)
    # roll1_receiver = roll1_particle[r1_mask]
    # roll1_pos = minPoint[roll1_receiver, r1_min_indices[roll1_receiver],:]

    # r2_mask = torch.isin(roll2_particle, particle_indices)
    # roll2_receiver = roll2_particle[r2_mask]
    # roll2_pos = minPoint[roll2_receiver,r2_min_indices[roll2_receiver],:]

    receiver = torch.hstack([hopper_receiver, roll1_receiver, roll2_receiver])
    pos = torch.vstack([hopper_pos, roll1_pos, roll2_pos])

    element_idx = torch.hstack([hopper_sender, roll1_sender, roll2_sender])
    vel = torch.mean(mesh_vel[cells[element_idx],:], dim=1)

    num_hop = hopper_receiver.shape[0]
    num_roll1 = roll1_receiver.shape[0]
    num_roll2 = roll2_receiver.shape[0]
    contact_type = torch.tensor([1 for _ in range(num_hop)] + [2 for _ in range(num_roll1)]+[3 for _ in range(num_roll2)], device=mesh_pos.device)

    if pos.ndim == 1:
        pos = pos.expand(1,3)
        receiver = receiver.expand(1)
        vel = vel.expand(1,3)
        norm = norm.expand(1,6)

    return receiver, pos, vel, contact_type, norm

"""

if __name__ == "__main__":
    print("\n===== EDGE LOCAL FRAME DEBUG START =====")

    # ======================
    # 1. dummy data 생성
    # ======================
    N = 10  # node 수
    E = 20  # edge 수

    pos = torch.randn(N, 3)
    vel = torch.randn(N, 3)

    receivers = torch.randint(0, N, (E,))
    senders = torch.randint(0, N, (E,))

    # self-edge 제거
    mask = receivers != senders
    receivers = receivers[mask]
    senders = senders[mask]

    print("num_edges:", receivers.shape[0])

    # ======================
    # 2. edge frame 계산
    # ======================
    out = build_edge_local_frame_3d(
        pos,
        vel,
        receivers,
        senders
    )

    a = out['a']
    b = out['b']
    c = out['c']
    rev = out['reverse_edge_idx']

    # ======================
    # 3. shape 확인
    # ======================
    print("\n[Shape check]")
    print("a:", a.shape)
    print("b:", b.shape)
    print("c:", c.shape)

    # ======================
    # 4. NaN 체크
    # ======================
    print("\n[NaN check]")
    print("a NaN:", torch.isnan(a).any().item())
    print("b NaN:", torch.isnan(b).any().item())
    print("c NaN:", torch.isnan(c).any().item())

    # ======================
    # 5. norm 확인 (단위벡터인지)
    # ======================
    print("\n[Norm check]")
    print("a norm mean:", torch.norm(a, dim=1).mean().item())
    print("b norm mean:", torch.norm(b, dim=1).mean().item())
    print("c norm mean:", torch.norm(c, dim=1).mean().item())

    # ======================
    # 6. 직교성 확인
    # ======================
    print("\n[Orthogonality check]")
    ab = torch.sum(a * b, dim=1).abs().mean().item()
    ac = torch.sum(a * c, dim=1).abs().mean().item()
    bc = torch.sum(b * c, dim=1).abs().mean().item()

    print("a·b:", ab)
    print("a·c:", ac)
    print("b·c:", bc)

    # ======================
    # 7. antisymmetry 체크
    # ======================
    print("\n[Antisymmetry check]")

    valid = rev >= 0
    if valid.sum() > 0:
        a_sym = (a[valid] + a[rev[valid]]).abs().mean().item()
        b_sym = (b[valid] + b[rev[valid]]).abs().mean().item()
        c_sym = (c[valid] + c[rev[valid]]).abs().mean().item()

        print("a_ij + a_ji:", a_sym)
        print("b_ij + b_ji:", b_sym)
        print("c_ij + c_ji:", c_sym)
    else:
        print("No reverse edges found")

    # ======================
    # 8. degenerate 체크
    # ======================
    print("\n[Degenerate cases]")
    print("b degenerate:", out['b_degenerate_mask'].sum().item())
    print("c degenerate:", out['c_degenerate_mask'].sum().item())

    print("\n===== EDGE LOCAL FRAME DEBUG END =====")
    """