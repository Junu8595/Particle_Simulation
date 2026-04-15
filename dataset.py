import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from collections import namedtuple
from dataclasses import dataclass
from dataclasses import replace as dc_replace
import os
import graph_builder
from sklearn import neighbors
from torch_geometric.nn import radius_graph


@dataclass(frozen=False)
class RawDataPack:
    cells : torch.tensor

    prev_particle_pos : torch.tensor
    particle_pos : torch.tensor
    next_particle_pos : torch.tensor

    prev_mesh_pos : torch.tensor
    mesh_pos : torch.tensor
    next_mesh_pos : torch.tensor

    particle_vel : torch.tensor
    next_particle_vel : torch.tensor

    mesh_vel : torch.tensor
    next_mesh_vel : torch.tensor

    acc : torch.tensor
    next_acc : torch.tensor

    mesh_node_type : torch.tensor

    particle_id : torch.tensor
    next_particle_id : torch.tensor

    def todevice(self, device):
        for field in self.__dataclass_fields__:
            data = getattr(self, field)
            setattr(self, field, data.to(device))

NodePack = namedtuple(
    'NodePack', 
    ['node_features', 'particle_indices', 'next_particle_indices',
     'hopper_indices', 'roller1_indices', 'roller2_indices']
)

EdgePack = namedtuple(
    'EdgePack', 
    ['edge_features', 'receivers', 'senders', 'edge_a', 'edge_b', 'edge_c', 
     'reverse_edge_idx', 'pairwise_mask', 'b_degenerate_mask', 'c_degenerate_mask']
)

TargetPack = namedtuple(
    'TargetPack', 
    ['normalized_target', 'target_acc', 'target_vel', 'target_pos']
)

DataPack = namedtuple('DataPack', ['nodepack', 'edgepack', 'targetpack'])

class gns_dataset(Dataset):
    def __init__(self, dataset_properties_packs, normalizer_packs, device, mode='train'):

        self.maximum_mesh_edges = 0
        self.maximum_particle_edges = 0
        self.maximum_mesh_edges_node = 0
        self.maximum_particle_edges_node = 0

        self.device = device
        self.ds_path = dataset_properties_packs.ds_path
        self.mode = mode # 'train' 또는 'test'

        self.training_folder = dataset_properties_packs.training_path
        self.test_folder = dataset_properties_packs.testing_path
        self.noise_level = dataset_properties_packs.training_noise

        self.contact_distance = dataset_properties_packs.contact_distance

        self.node_normalizer = normalizer_packs[0]
        self.edge_normalizer = normalizer_packs[1]
        self.target_normalizer = normalizer_packs[2]

        self.num_history = dataset_properties_packs.num_history
        self.lower = dataset_properties_packs.num_history
        self.upper = -2

        self.bake_mode = False  # True일 때: normalizer/log-transform 생략, raw feature 저장

        self.node_type_dict = {'particle' : int(0),
                               'hopper' : int(1),
                               'roller1' : int(2),
                               'roller2' : int(3)}
        
    def load_dataset(self, ds_path):
        self.dataset = self.read_dataset(ds_path)
        self.iterator = self.set_iterator(self.dataset)
    
    def read_dataset(self, ds_path): #데이터 읽을때 숫자 외 텍스트로 시작하는 파일은 무시

        ds_list = os.listdir(ds_path)

        valid_prefix = []
        for file in ds_list:
            if not file.endswith('.npy'):
                continue

            prefix = file.split('_')[0]
            if prefix.isdigit():
                valid_prefix.append(int(prefix))

        ds_list = np.unique(np.array(valid_prefix).astype(int))

        records = []

        for idx in ds_list:

            particle_pos = np.load(os.path.join(ds_path, str(idx).zfill(5) + '_particle_positions.npy'))
            particle_pos = torch.from_numpy(particle_pos).float().to(device='cpu') * 1000

            mesh_pos = np.load(os.path.join(ds_path, str(idx).zfill(5) + '_geo_pos.npy'))
            element_cnt = int(mesh_pos.shape[1])
            mesh_pos = torch.from_numpy(mesh_pos).float().to(device='cpu')[1:] * 1000

            cells = np.load(os.path.join(ds_path, str(idx).zfill(5) + '_geo_ele.npy'))
            cells = torch.from_numpy(cells).long().to(device='cpu')

            mesh_node_type = np.load(os.path.join(ds_path, str(idx).zfill(5) + '_geo_node_type.npy'))
            mesh_node_type = torch.from_numpy(mesh_node_type).long().to(device='cpu')

            particle_id = np.load(os.path.join(ds_path, str(idx).zfill(5) + '_particle_id.npy'))
            particle_id = torch.from_numpy(particle_id).long().to(device='cpu')

            records.append([cells, mesh_pos, particle_pos, mesh_node_type, particle_id])

        return records
        
    def set_iterator(self, dataset):
        idx_list = []

        for i in range(len(dataset)):
            lower = self.lower
            upper = self.dataset[i][2].shape[0] + self.upper

            for j in range(lower, upper):
                idx_list.append([i,j])

        return np.array(idx_list)
    
    def get_data(self, idx, contact_distance, rotate_flag):
        with torch.no_grad():
            raw_data = self.get_raw_data(idx)
            # ✅ GPU로 보내는 코드를 제거하고, CPU 상태로 graph_data를 만듭니다.
            return self.graph_data(contact_distance, raw_data, rotate_flag)

    def get_raw_data(self, idx):

        file_idx = self.iterator[idx, 0]
        iter_idx = self.iterator[idx, 1]

        self.file_idx = file_idx
        self.iter_idx = iter_idx

        prev_iter_idx = max(iter_idx-1, 0)
        next_iter_idx = iter_idx + 1
        prev_vel_iter_idx = max(iter_idx+1-self.num_history, 0)

        datapack = self.dataset[file_idx]

        # 모든 데이터를 CPU로 이동
        cells_data = datapack[0].to('cpu') # (N_E, 3)
        mesh_pos_data = datapack[1].to('cpu')
        particle_pos_data = datapack[2].to('cpu') # (T, N, 3)
        mesh_node_type_data = datapack[3].to('cpu') # (T, N_E)
        particle_id_data = datapack[4].to('cpu') # (T, N_P) particle이 domain내에 존재하는지 안하는지

        cells_indicies = torch.tensor(range(cells_data.shape[0]), device='cpu')
        cells = torch.hstack((cells_indicies[:, None], cells_data))

        mesh_node_type = mesh_node_type_data.clone().float()

        particle_id = particle_id_data[iter_idx]
        next_particle_id = particle_id_data[next_iter_idx]

        particle_indices = particle_id.nonzero().squeeze()
        next_particle_indices = next_particle_id.nonzero().squeeze()

        prev_particle_pos = particle_pos_data[prev_iter_idx].clone().float()
        particle_pos = particle_pos_data[iter_idx].clone().float()
        next_particle_pos = particle_pos_data[next_iter_idx].clone().float()

        prev_mesh_pos = mesh_pos_data[prev_iter_idx].clone().float()
        mesh_pos = mesh_pos_data[iter_idx].clone().float()
        next_mesh_pos = mesh_pos_data[next_iter_idx].clone().float()

        if self.noise_level > 1e-8:
            noise = (torch.randn(particle_pos.shape) * self.noise_level).to(particle_pos.to('cpu'))
            particle_pos[particle_indices] += noise[particle_indices]

        particle_vel = (particle_pos_data[prev_vel_iter_idx:iter_idx+1]-particle_pos_data[prev_vel_iter_idx-1:iter_idx]).clone().float()
        particle_vel = particle_vel.permute(1,0,2).reshape(particle_pos_data.shape[1],-1)
        next_particle_vel = (particle_pos_data[next_iter_idx]-particle_pos_data[iter_idx]).clone().float()

        mesh_vel = (mesh_pos_data[prev_vel_iter_idx:iter_idx+1]-mesh_pos_data[prev_vel_iter_idx-1:iter_idx]).clone().float()
        mesh_vel = mesh_vel.permute(1,0,2).reshape(mesh_pos_data.shape[1], -1)
        next_mesh_vel = (mesh_pos_data[next_iter_idx]-mesh_pos_data[iter_idx]).clone().float()

        acc = particle_vel[:,-3:] - particle_vel[:,-6:-3]
        next_acc = next_particle_vel - particle_vel[:,-3:]

        return RawDataPack(cells, prev_particle_pos, particle_pos, next_particle_pos, prev_mesh_pos, mesh_pos, next_mesh_pos, particle_vel, next_particle_vel,
                           mesh_vel, next_mesh_vel, acc, next_acc, mesh_node_type, particle_id, next_particle_id)
    
    def graph_data(self, contact_distance, data, rotate_flag):

        self_edge = False

        cells = data.cells
        mesh_node_type = data.mesh_node_type.float()

        particle_pos = data.particle_pos.float()
        next_particle_pos = data.next_particle_pos.float()

        mesh_pos = data.mesh_pos.float()

        particle_vel = data.particle_vel.float()
        next_particle_vel = data.next_particle_vel.float()

        mesh_vel = data.mesh_vel.float()

        particle_id = data.particle_id.float()
        next_particle_id = data.next_particle_id.float()

        particle_indices = particle_id.nonzero(as_tuple=False).reshape(-1)
        next_particle_indices = next_particle_id.nonzero(as_tuple=False).reshape(-1)

        self.particle_indices = particle_indices
        self.next_particle_indices = next_particle_indices
        num_nodes = particle_pos.shape[0]
        sub_indices = particle_indices
        sub_particle_pos = particle_pos[sub_indices]

        if sub_particle_pos.ndim == 1:
            sub_particle_pos = sub_particle_pos[None, :]   # (1,3)

        if sub_particle_pos.shape[0] != 0:
            
            # 1. radius_graph를 이용한 탐색
            # sub_particle_pos가 GPU에 있다면 GPU 연산을, CPU에 있다면 최적화된 C++ CPU 연산을 수행합니다.
            edge_index = radius_graph(
                sub_particle_pos,                
                r=contact_distance,           
                batch=None,       
                loop=False,         # 자기 자신(Self-loop) 연결 제외 (self_edge=False와 동일 효과)
                max_num_neighbors=64 # 물리 시뮬레이션 밀도에 맞게 설정 (기본값 32)
            )

            # 2. edge_index 분리 [2, num_edges]
            senders_sub = edge_index[0]   # 첫 번째 행: Senders
            receivers_sub = edge_index[1] # 두 번째 행: Receivers

            senders = sub_indices[senders_sub]
            receivers = sub_indices[receivers_sub]
        else:
            senders = torch.empty((0,), dtype=sub_indices.dtype, device='cpu')
            receivers = torch.empty((0,), dtype=sub_indices.dtype, device='cpu')

        if not self_edge:
            mask = receivers != senders
            receivers = receivers[mask]
            senders = senders[mask]

        edges = torch.vstack((receivers, senders)).unique(dim=1)

        receivers = edges[0]
        senders = edges[1]

        if self.maximum_particle_edges < receivers.shape[0]:
            self.maximum_particle_edges = receivers.shape[0]
            self.maximum_particle_edges_node = num_nodes

        norm = torch.tensor([0, 0, 0, 0, 0, 0]).float().to('cpu')
        norm = norm.repeat(num_nodes, 1).to('cpu')

        if mesh_pos.shape[0] != 0:
            mesh_receivers, contact_pos, contact_vel, contact_type, mesh_norm = graph_builder.build_boundary_edge(cells[:,1:], particle_pos, mesh_pos, particle_indices, mesh_node_type, mesh_vel, 'cpu', contact_distance)

        else:
            mesh_receivers = torch.empty((0), device='cpu', dtype = receivers.dtype)
            contact_pos = torch.empty((0,particle_pos.shape[1]), device='cpu', dtype=particle_pos.dtype)
            contact_vel = torch.empty((0,particle_vel.shape[1]), device='cpu', dtype=particle_vel.dtype)
            contact_type = torch.empty((0), device='cpu', dtype=torch.int32)
            mesh_norm = torch.empty((0, norm.shape[1]), device='cpu', dtype=norm.dtype)

        if self.maximum_mesh_edges < mesh_receivers.shape[0]:
            self.maximum_mesh_edges = mesh_receivers.shape[0]
            self.maximum_mesh_edges_node = num_nodes

        norm = torch.vstack([norm, mesh_norm])
        
        num_pairwise_edges = receivers.shape[0] # contact edge를 붙이기 전 particle-particle edge 개수 저장
        
        receivers = torch.hstack([receivers,mesh_receivers])
        senders = torch.hstack([senders,torch.tensor([i for i in range(particle_pos.shape[0], particle_pos.shape[0]+contact_pos.shape[0])], device='cpu', dtype=torch.int64)])

        pos = torch.vstack([particle_pos, contact_pos])
        next_pos = torch.vstack([next_particle_pos, contact_pos])

        vel = torch.vstack([particle_vel, contact_vel])
        next_vel = torch.vstack([next_particle_vel, contact_vel[:,-3:]])

        node_type = torch.hstack([torch.tensor([0 for _ in range(particle_pos.shape[0])], device='cpu'), contact_type]).to('cpu')
        node_type = graph_builder.build_node_type(node_type)

        hopper = self.node_type_dict['hopper']
        roller1 = self.node_type_dict['roller1']
        roller2 = self.node_type_dict['roller2']

        hopper_indices = (node_type[:, hopper] == 1).nonzero(as_tuple=False).reshape(-1)
        roller1_indices = (node_type[:, roller1] == 1).nonzero(as_tuple=False).reshape(-1)
        roller2_indices = (node_type[:, roller2] == 1).nonzero(as_tuple=False).reshape(-1)

        # if rotate_flag:

        #     x_rot_angle = torch.rand((1)) * 2 * np.pi
        #     y_rot_angle = torch.rand((1)) * 2 * np.pi
        #     z_rot_angle = torch.rand((1)) * 2 * np.pi

        #     x_rot_mat = graph_builder.make_rotation_mat(pos, x_rot_angle, 'x')
        #     y_rot_mat = graph_builder.make_rotation_mat(pos, y_rot_angle, 'y')
        #     z_rot_mat = graph_builder.make_rotation_mat(pos, z_rot_angle, 'z')

        #     pos = graph_builder.rotate_pos(pos, x_rot_mat)
        #     pos = graph_builder.rotate_pos(pos, y_rot_mat)
        #     pos = graph_builder.rotate_pos(pos, z_rot_mat)

        #     next_pos = graph_builder.rotate_pos(next_pos, x_rot_mat)
        #     next_pos = graph_builder.rotate_pos(next_pos, y_rot_mat)
        #     next_pos = graph_builder.rotate_pos(next_pos, z_rot_mat)

        target_acc = (next_vel - vel[:,-3:]).detach()
        target_vel = next_vel.detach()
        target_pos = next_pos.detach()

        if not self.bake_mode:
            normalized_target = self.target_normalizer.forward(target_acc[:particle_indices.shape[0]].to('cpu'), accumulate=True)
            normalized_target = self.target_normalizer.forward(target_acc.to('cpu'), accumulate=False).detach().to('cpu').clone()
            normalized_target_sign = torch.where(normalized_target >= 0.0, 1, -1)
            normalized_target = torch.log(normalized_target.abs() + 1) * normalized_target_sign
        else:
            # bake_mode: raw target_acc 그대로 저장 (normalization/log-transform 생략)
            normalized_target = target_acc.detach().to('cpu').clone()

        # =========================================================
        # Edge local frame for particle-particle edges only
        # particle-particle edge에 대해서만 3D edge local frame 계산
        # contact edge는 reverse edge가 없을 수 있으므로 0으로 padding
        # 이후 EdgePack에 함께 담아서 graph_model로 전달
        # =========================================================
        
        num_total_edges = receivers.shape[0]
        pairwise_mask = torch.zeros(num_total_edges, dtype=torch.bool, device='cpu')
        pairwise_mask[:num_pairwise_edges] = True

        edge_a = torch.zeros((num_total_edges, 3), dtype=pos.dtype, device='cpu')
        edge_b = torch.zeros((num_total_edges, 3), dtype=pos.dtype, device='cpu')
        edge_c = torch.zeros((num_total_edges, 3), dtype=pos.dtype, device='cpu')

        reverse_edge_idx = torch.full((num_total_edges,), -1, dtype=torch.long, device='cpu')
        b_degenerate_mask = torch.zeros((num_total_edges,), dtype=torch.bool, device='cpu')
        c_degenerate_mask = torch.zeros((num_total_edges,), dtype=torch.bool, device='cpu')
                
        if num_pairwise_edges > 0:
            pair_receivers = receivers[:num_pairwise_edges]
            pair_senders = senders[:num_pairwise_edges]
            (
            edge_a_pw,
            edge_b_pw,
            edge_c_pw,
            reverse_edge_idx_pw,
            b_degenerate_mask_pw,
            c_degenerate_mask_pw,
            ) = graph_builder.build_edge_local_frame_3d(
            pos[:particle_pos.shape[0]],
            vel[:particle_pos.shape[0], -3:],
            pair_receivers,
            pair_senders
            )
            edge_a[:num_pairwise_edges] = edge_a_pw
            edge_b[:num_pairwise_edges] = edge_b_pw
            edge_c[:num_pairwise_edges] = edge_c_pw

            reverse_edge_idx[:num_pairwise_edges] = reverse_edge_idx_pw
            b_degenerate_mask[:num_pairwise_edges] = b_degenerate_mask_pw
            c_degenerate_mask[:num_pairwise_edges] = c_degenerate_mask_pw

        # === Issue #1 Fix: Contact edge local frame ===
        # PP 엣지와 달리 Contact(PM) 엣지는 역방향 엣지가 없으므로,
        # mesh face normal을 이용해 직교 프레임을 구성한다.
        num_contact_edges = num_total_edges - num_pairwise_edges
        if num_contact_edges > 0:
            # a: 입자에서 접촉점 방향
            contact_rel_pos = pos[senders[num_pairwise_edges:]] - pos[receivers[num_pairwise_edges:]]
            contact_a = graph_builder.safe_normalize(contact_rel_pos)

            # b: face normal에서 a 성분 제거 (Gram-Schmidt)
            contact_normal = mesh_norm[:, :3]  # build_boundary_edge가 반환한 face normal
            a_dot_n = torch.sum(contact_normal * contact_a, dim=1, keepdim=True)
            normal_perp = contact_normal - a_dot_n * contact_a
            contact_b = graph_builder.safe_normalize(normal_perp)

            # degenerate: |normal_perp| < 1e-4 이면 GS가 수치적으로 불안정
            # (face normal이 contact_a와 거의 평행할 때 float32 오차로 방향이 왜곡됨)
            b_degen = torch.norm(normal_perp, dim=1) < 1e-4
            if b_degen.any():
                contact_b[b_degen] = graph_builder.build_fallback_b_from_a(contact_a[b_degen])

            # c: a x b
            contact_c = graph_builder.safe_normalize(torch.cross(contact_a, contact_b, dim=-1))

            edge_a[num_pairwise_edges:] = contact_a
            edge_b[num_pairwise_edges:] = contact_b
            edge_c[num_pairwise_edges:] = contact_c

        # relative position / velocity
        rel_pos = pos[senders] - pos[receivers]                 # (E, 3)
        dist = torch.norm(rel_pos, dim=1, keepdim=True)         # (E, 1)

        cur_vel = vel[:, -3:]                                   # (N_total, 3)
        rel_vel = cur_vel[senders] - cur_vel[receivers]         # (E, 3)

        dx_local = torch.zeros((num_total_edges, 3), dtype=pos.dtype, device='cpu')
        dv_local = torch.zeros((num_total_edges, 3), dtype=pos.dtype, device='cpu')

        if pairwise_mask.any():
            pw = pairwise_mask

            dx_local[pw, 0] = torch.sum(rel_pos[pw] * edge_a[pw], dim=1)
            dx_local[pw, 1] = torch.sum(rel_pos[pw] * edge_b[pw], dim=1)
            dx_local[pw, 2] = torch.sum(rel_pos[pw] * edge_c[pw], dim=1)

            dv_local[pw, 0] = torch.sum(rel_vel[pw] * edge_a[pw], dim=1)
            dv_local[pw, 1] = torch.sum(rel_vel[pw] * edge_b[pw], dim=1)
            dv_local[pw, 2] = torch.sum(rel_vel[pw] * edge_c[pw], dim=1)

        # === Issue #2 Fix: Contact edge feature projection ===
        # #1 수정 후 contact 엣지에도 유효한 local frame이 있으므로
        # PP와 동일하게 rel_pos / rel_vel을 local frame에 project한다.
        cm = ~pairwise_mask  # contact mask
        if cm.any():
            dx_local[cm, 0] = torch.sum(rel_pos[cm] * edge_a[cm], dim=1)
            dx_local[cm, 1] = torch.sum(rel_pos[cm] * edge_b[cm], dim=1)
            dx_local[cm, 2] = torch.sum(rel_pos[cm] * edge_c[cm], dim=1)

            dv_local[cm, 0] = torch.sum(rel_vel[cm] * edge_a[cm], dim=1)
            dv_local[cm, 1] = torch.sum(rel_vel[cm] * edge_b[cm], dim=1)
            dv_local[cm, 2] = torch.sum(rel_vel[cm] * edge_c[cm], dim=1)

      # 최종 scalarized edge feature: 7차원
        edge_features = torch.hstack([dist, dx_local, dv_local])
        if not self.bake_mode:
            edge_features = self.edge_normalizer(edge_features.to('cpu'), accumulate=True).to('cpu')
        else:
            edge_features = edge_features.to('cpu')

        # node feature
        node_features = torch.hstack((vel, node_type, norm))
        if not self.bake_mode:
            node_features = self.node_normalizer(node_features.to('cpu'), accumulate=True).to('cpu')
        else:
            node_features = node_features.to('cpu')

        nodepack = NodePack(
            node_features,
            particle_indices,
            next_particle_indices,
            hopper_indices,
            roller1_indices,
            roller2_indices
         )

        
    
        edgepack = EdgePack(
            edge_features,
         receivers,
         senders,
         edge_a,
         edge_b,
         edge_c,
         reverse_edge_idx,
         pairwise_mask,
         b_degenerate_mask,
         c_degenerate_mask
         )

        
        targetpack = TargetPack(
         normalized_target,
         target_acc,
         target_vel,
         target_pos
         )

        datapack = DataPack(nodepack, edgepack, targetpack)

        return datapack
    
    def project_to_local_frame(self, vec, edge_a, edge_b, edge_c):
     """
     vec:    (E, 3)
     edge_a: (E, 3)
     edge_b: (E, 3)
     edge_c: (E, 3)
     return: (E, 3) = [vec·a, vec·b, vec·c]
     """
     s_a = torch.sum(vec * edge_a, dim=1, keepdim=True)
     s_b = torch.sum(vec * edge_b, dim=1, keepdim=True)
     s_c = torch.sum(vec * edge_c, dim=1, keepdim=True)
     return torch.hstack([s_a, s_b, s_c])


    def build_scalarized_edge_features(self, pos, vel, receivers, senders, edge_a, edge_b, edge_c, pairwise_mask):
     """
     진행상황3 최소 구현:
     - pairwise edge: dx, dv를 local frame에 projection
     - contact edge : projection 대신 0-padding
     - 최종 feature: [dist, dx_a, dx_b, dx_c, dv_a, dv_b, dv_c]  -> 7차원
     """

     rel_pos = pos[senders] - pos[receivers]          # (E, 3)
     dist = torch.norm(rel_pos, dim=1, keepdim=True)  # (E, 1)

     # 최근 velocity 3성분만 사용
     cur_vel = vel[:, -3:]                            # (N, 3)
     rel_vel = cur_vel[senders] - cur_vel[receivers]  # (E, 3)
 
     # 기본값 0
     dx_local = torch.zeros((rel_pos.shape[0], 3), device='cpu', dtype=rel_pos.dtype)
     dv_local = torch.zeros((rel_vel.shape[0], 3), device='cpu', dtype=rel_vel.dtype)

     if pairwise_mask.any():
        pw = pairwise_mask
        dx_local[pw] = self.project_to_local_frame(
            rel_pos[pw], edge_a[pw], edge_b[pw], edge_c[pw]
        )
        dv_local[pw] = self.project_to_local_frame(
            rel_vel[pw], edge_a[pw], edge_b[pw], edge_c[pw]
        )

     edge_features = torch.hstack([dist, dx_local, dv_local])  # (E, 7)
     return edge_features
    
    def __len__(self):
        return self.iterator.shape[0]
    
    def shuffle(self):
        np.random.shuffle(self.iterator)

    def return_target_sequence(self, raw_data):
        return torch.cat((raw_data.particle_pos, raw_data.particle_vel[:,-3:], raw_data.acc, raw_data.particle_id.unsqueeze(-1)), dim=1).clone()
    
    def data_from_test_set(self, raw_data_pack):
        updated_vel = raw_data_pack.particle_vel.clone()
        updated_prev_pos = raw_data_pack.prev_particle_pos.clone()
        updated_pos = raw_data_pack.particle_pos.clone()
        updated_acc = raw_data_pack.acc.clone()

        return updated_vel, updated_prev_pos, updated_pos, updated_acc
    
    def update_raw_data(self, raw_data_pack, updated_vel, updated_prev_pos, updated_pos, updated_acc):
        raw_data_pack = dc_replace(raw_data_pack, particle_vel = updated_vel.clone().to('cpu'))
        raw_data_pack = dc_replace(raw_data_pack, prev_particle_pos = updated_prev_pos.clone().to('cpu'))
        raw_data_pack = dc_replace(raw_data_pack, particle_pos = updated_pos.clone().to('cpu'))
        raw_data_pack = dc_replace(raw_data_pack, acc = updated_acc.clone().to('cpu'))
        return raw_data_pack
    
    def update_data(self, raw_data, contact_distance):
        with torch.no_grad():
            raw_data.todevice('cpu')
            return self.graph_data(contact_distance, raw_data, False)
        
    def reverse_output(self, output, updated_pos, updated_vel):
        output = output.to('cpu')        
        output_sign = torch.where(output >= 0.0, 1, -1)
        output = torch.clamp(output.abs(), max=10.0)
        output = (torch.pow(np.e * torch.ones(output.shape, device='cpu'), output) - 1) * output_sign
        acc_prediction = self.target_normalizer.inverse(output)
        
        updated_pos = updated_pos.to('cpu')
        updated_vel = updated_vel.to('cpu')

        vel_prediction = torch.zeros_like(updated_vel).float().to('cpu')
        vel_prediction[:, :-3] = updated_vel[:, 3:]
        vel_prediction[:, -3:] = updated_vel[:, -3:] + acc_prediction
        pos_prediction = updated_pos + vel_prediction[:, -3:]

        return pos_prediction, vel_prediction, acc_prediction
    
    def update_test_data(self, raw_data_pack, updated_prev_pos, updated_pos, updated_vel, updated_acc, pos_prediction, vel_prediction, acc_prediction):

        updated_prev_pos = updated_pos[:raw_data_pack.particle_id.shape[0],:].detach().cpu().clone()

        updated_pos = pos_prediction.detach().cpu().clone()

        updated_vel = vel_prediction.detach().cpu().clone()

        updated_acc = acc_prediction.detach().cpu().clone()

        return updated_prev_pos, updated_pos, updated_vel, updated_acc
    
    def set_sequence(self, idx):
        mask = np.where(self.iterator[:,0] == idx, 1, 0).nonzero()

        self.iterator = self.iterator[mask]
        self.iterator = self.iterator[self.iterator[:,1].argsort()]

    def compute_xyz_bounds(self, raw_data_pack):
        particle_indices = raw_data_pack.particle_id.nonzero(as_tuple=False).reshape(-1)

        particle_position = raw_data_pack.particle_pos[particle_indices].detach().cpu()

        if particle_position.numel() == 0:
            return 0, 0, 0, 0, 0, 0
        
        xmin = float(torch.min(particle_position[:,0]))
        xmax = float(torch.max(particle_position[:,0]))

        ymin = float(torch.min(particle_position[:,1]))
        ymax = float(torch.max(particle_position[:,1]))

        zmin = float(torch.min(particle_position[:,2]))
        zmax = float(torch.max(particle_position[:,2]))

        return xmin, xmax, ymin, ymax, zmin, zmax
    
    def box_mask_xyz(self, position, xbound, ybound, zbound):
        x, y, z = position[:,0], position[:,1], position[:,2]

        return (x >= xbound[0]) & (x <= xbound[1]) & (y >= ybound[0]) & (y <= ybound[1]) & (z >= zbound[0]) & (z <= zbound[1])

    def get_intersecting_mesh_mask(self, raw_data_pack, xbound, ybound, zbound):
        cell_node_indices = raw_data_pack.cells[:, 1:].to('cpu')
        cell_pos = raw_data_pack.mesh_pos[cell_node_indices].to('cpu') 

        cell_min = torch.min(cell_pos, dim=1)[0] # (N_cells, 3)
        cell_max = torch.max(cell_pos, dim=1)[0] # (N_cells, 3)

        grid_min = torch.tensor([xbound[0], ybound[0], zbound[0]], device='cpu')
        grid_max = torch.tensor([xbound[1], ybound[1], zbound[1]], device='cpu')

        intersect_x = (cell_max[:, 0] >= grid_min[0]) & (cell_min[:, 0] <= grid_max[0])
        intersect_y = (cell_max[:, 1] >= grid_min[1]) & (cell_min[:, 1] <= grid_max[1])
        intersect_z = (cell_max[:, 2] >= grid_min[2]) & (cell_min[:, 2] <= grid_max[2])

        cell_mask = intersect_x & intersect_y & intersect_z # (N_cells, )

        valid_cells = raw_data_pack.cells[cell_mask]
        valid_node_indices = torch.unique(valid_cells[:, 1:])

        num_mesh_nodes = raw_data_pack.mesh_pos.shape[0]
        mesh_mask = torch.zeros(num_mesh_nodes, dtype=torch.bool, device='cpu')
        mesh_mask[valid_node_indices] = True

        return mesh_mask
    
    def reindex_cells(self, old_to_new, cells):
        out = cells.clone()
        node_ids = out[:, 1:].reshape(-1).detach().cpu().tolist()
        mapped = [old_to_new[int(v)] for v in node_ids]
        out[:, 1:] = torch.tensor(mapped, dtype=torch.long, device='cpu').view_as(out[:, 1:])

        return out
    
    def slice_datapack(self, raw_data_pack, particle_mask, mesh_mask):

        particle_mask = particle_mask & raw_data_pack.particle_id.bool()
        
        particle_idx = particle_mask.nonzero(as_tuple=False).reshape(-1)
        mesh_idx = mesh_mask.nonzero(as_tuple=False).reshape(-1)

        prev_particle_pos = raw_data_pack.prev_particle_pos[particle_idx].clone()
        particle_pos = raw_data_pack.particle_pos[particle_idx].clone()
        next_particle_pos = raw_data_pack.next_particle_pos[particle_idx].clone()

        particle_vel = raw_data_pack.particle_vel[particle_idx].clone()
        next_particle_vel = raw_data_pack.next_particle_vel[particle_idx].clone()

        acc = raw_data_pack.acc[particle_idx].clone()
        next_acc = raw_data_pack.next_acc[particle_idx].clone()

        particle_id = raw_data_pack.particle_id[particle_idx].clone()
        next_particle_id = raw_data_pack.next_particle_id[particle_idx].clone()

        keep_cell = mesh_mask[raw_data_pack.cells[:,1:]].any(dim=1)
        cells = raw_data_pack.cells[keep_cell]

        new_mesh_node = torch.unique(cells[:,1:].reshape(-1))

        old_to_new = {int (i) : idx for idx, i in enumerate(new_mesh_node.detach().cpu().tolist())}

        cells = self.reindex_cells(old_to_new, cells)

        prev_mesh_pos = raw_data_pack.prev_mesh_pos[new_mesh_node]
        mesh_pos = raw_data_pack.mesh_pos[new_mesh_node]
        mesh_node_type = raw_data_pack.mesh_node_type[new_mesh_node]
        next_mesh_pos = raw_data_pack.next_mesh_pos[new_mesh_node]

        mesh_vel = raw_data_pack.mesh_vel[new_mesh_node]
        next_mesh_vel = raw_data_pack.next_mesh_vel[new_mesh_node]

        return RawDataPack(cells, prev_particle_pos, particle_pos, next_particle_pos, prev_mesh_pos, mesh_pos, next_mesh_pos, particle_vel, next_particle_vel, mesh_vel, next_mesh_vel, acc, next_acc, mesh_node_type, particle_id, next_particle_id), particle_idx
    
    def build_tiled_raw_data(self, raw_data_pack, xbound, ybound, zbound, contact_distance):

        xbound_pad = (xbound[0] - contact_distance, xbound[1] + contact_distance)
        ybound_pad = (ybound[0] - contact_distance, ybound[1] + contact_distance)
        zbound_pad = (zbound[0] - contact_distance, zbound[1] + contact_distance)

        particle_mask = self.box_mask_xyz(raw_data_pack.particle_pos, xbound, ybound, zbound_pad)
        mesh_mask = self.get_intersecting_mesh_mask(raw_data_pack, xbound_pad, ybound_pad, zbound_pad)

        return self.slice_datapack(raw_data_pack, particle_mask, mesh_mask)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            baked_file_path = f'baked_training_data/step_{idx}.pt'
            data_pack = torch.load(baked_file_path, weights_only=False)

            # baked .pt 파일은 raw(비정규화) feature를 담고 있음.
            # 여기서 training normalizer를 적용하고 log-transform을 수행한다.
            node_features = self.node_normalizer(data_pack.nodepack.node_features, accumulate=True)
            edge_features = self.edge_normalizer(data_pack.edgepack.edge_features, accumulate=True)

            # targetpack.normalized_target에는 raw target_acc가 저장되어 있음 (bake_mode=True로 구운 경우)
            raw_target = data_pack.targetpack.normalized_target
            particle_idx = data_pack.nodepack.particle_indices
            # particle 노드만으로 통계 누적, 전체 노드에 적용
            _ = self.target_normalizer.forward(raw_target[:particle_idx.shape[0]], accumulate=True)
            normalized_target = self.target_normalizer.forward(raw_target, accumulate=False).detach()
            nt_sign = torch.where(normalized_target >= 0.0, 1, -1)
            normalized_target = torch.log(normalized_target.abs() + 1) * nt_sign

            new_np = data_pack.nodepack._replace(node_features=node_features)
            new_ep = data_pack.edgepack._replace(edge_features=edge_features)
            new_tp = data_pack.targetpack._replace(normalized_target=normalized_target)
            return DataPack(new_np, new_ep, new_tp)
        else:
            return self.get_data(idx, self.contact_distance, False)