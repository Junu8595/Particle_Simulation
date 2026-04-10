import torch
from torch.utils.data import DataLoader
import numpy as np

import os
from datetime import datetime

import copy
from torch.utils.data import DataLoader
import graph_model as gm
import dataset
import graph_utils as utils
import normalizer
import attributes as attr

# from IPython import get_ipython
# from IPython.display import clear_output

import matplotlib.pyplot as plt

from dataclasses import replace as dc_replace
from dataset import DataPack, NodePack, EdgePack, TargetPack

import cv2
import random

seed = 7777
torch.backends.cudnn.benchmark = True
torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = True   
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

train_flag = True
one_step_flag = True
roll_out_flag = False
rotate_flag = False
fresh_start = True
plot_flag = False

def mem(tag=""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated()/1024**3
        r = torch.cuda.memory_reserved()/1024**3
        m = torch.cuda.max_memory_allocated()/1024**3
        print(f"[{tag}] alloc={a:.2f}GB reserved={r:.2f}GB max_alloc={m:.2f}GB")

test_loss_list = []

device = torch.device('cuda:0') 
torch.cuda.set_device(device)

X_BOUND = [-50, 50]
Y_BOUND = [-50, 50]
Z_BOUND = [0, 100]

# torch.set_default_tensor_type('torch.cuda.FloatTensor')를 피함 - DataLoader sampler 충돌 방지
# 대신 dataset.py의 get_raw_data에서 모든 데이터를 명시적으로 device로 이동
network_attributes_pack, training_attributes_pack = attr.attribute(device)

training_parameters_pack, testing_parameters_pack, data_parameters_pack, optimizer_parameters_pack = training_attributes_pack

nepochs = training_parameters_pack.nepochs
latent_size = training_parameters_pack.latent_size
pre_accumulation_steps = training_parameters_pack.pre_accumulation_steps
message_passing_steps = training_parameters_pack.message_passing_steps

monitor_interval = testing_parameters_pack.monitor_interval
test_interval = testing_parameters_pack.test_interval
test_length = testing_parameters_pack.test_length
test_sequence_idx = testing_parameters_pack.test_sequence_idx

lr = optimizer_parameters_pack.lr
decay_offset = optimizer_parameters_pack.decay_offset
secondary_decay_offset = optimizer_parameters_pack.secondary_decay_offset
lr_decay_length = optimizer_parameters_pack.lr_decay_length
norm_acc_length = optimizer_parameters_pack.norm_acc_length
grad_limit = optimizer_parameters_pack.grad_limit

contact_distance = data_parameters_pack.contact_distance

time = utils.get_time()
load_path = './load_network/'
test_network_path = '/home/ssdl/PJW/Particle_Simulation/saves_2026_04_06_16_41_34/'
saving_path = './saves_' + time + '/'
test_result_path = './test_result_' + time + '/'

def get_balanced_overlapping_grids(particle_pos, indices, max_particles, min_particles, overlap_ratio=0.2, particle_mask = None):

    ABSOULUTE_PADDING = 4.0
    
    # 현재 구역의 입자 수
    num_particles = indices.shape[0]

    # 1. 종료 조건: 입자 수가 max_particles 이하이거나, 더 나누면 min_particles보다 작아질 경우
    def calculate_padded_box(subset_indices):
            
        # 현재 입자들을 감싸는 Bounding Box 계산
        pos_subset = particle_pos[subset_indices]
        min_xyz = torch.min(pos_subset, dim=0)[0]
        max_xyz = torch.max(pos_subset, dim=0)[0]
        
        xl, yl, zl = min_xyz[0].item(), min_xyz[1].item(), min_xyz[2].item()
        xr, yr, zr = max_xyz[0].item(), max_xyz[1].item(), max_xyz[2].item()

        # Overlap 적용: 박스 크기를 overlap_ratio 만큼 확장
        width_x, width_y, width_z = xr - xl, yr - yl, zr - zl
        
        # 0이 될 수 있는 width 방지 (입자가 1개일 경우 등)
        if width_x == 0: width_x = 0.1
        if width_y == 0: width_y = 0.1
        if width_z == 0: width_z = 0.1

        padding_x = max(width_x * overlap_ratio * 0.5, ABSOULUTE_PADDING)
        padding_y = max(width_y * overlap_ratio * 0.5, ABSOULUTE_PADDING)
        padding_z = max(width_z * overlap_ratio * 0.5, ABSOULUTE_PADDING)

        return (xl - padding_x, xr + padding_x, 
                 yl - padding_y, yr + padding_y, 
                 zl - padding_z, zr + padding_z)

    # 2. 분할 로직: 입자가 가장 넓게 퍼진 축을 찾음 (분산이 큰 축)

    if num_particles < 1:
        return []
    
    box = calculate_padded_box(indices)
    xl, xr, yl, yr, zl, zr = box

    px = particle_pos[:,0]
    py = particle_pos[:,1]
    pz = particle_pos[:,2]

    mask = (px >= xl) & (px < xr) & (py >= yl) & (py < yr) & (pz >= zl) & (pz < zr)
    if particle_mask is not None:
        mask = mask & particle_mask
    expanded_count = mask.sum().item()

    if expanded_count <= max_particles:
        return [box]

    pos_subset = particle_pos[indices]

    if num_particles == 1:
        return [box]
    
    # 각 축(x,y,z)의 범위(Range) 계산
    # min_vals = torch.min(pos_subset, dim=0)[0]
    # max_vals = torch.max(pos_subset, dim=0)[0]
    # ranges = max_vals - min_vals

    stds = torch.std(pos_subset, dim=0)
    
    axis = torch.argmax(stds).item()# 가장 긴 축 선택 (0:x, 1:y, 2:z)

    # 3. Median Split: 해당 축을 기준으로 입자들을 정렬하여 정확히 반으로 나눔
    # 이렇게 하면 양쪽 자식 노드는 비슷한 수의 입자를 가지게 됨
    values = pos_subset[:, axis]
    sorted_idx = torch.argsort(values)
    
    mid_idx = num_particles // 2

    if mid_idx == 0 or mid_idx == num_particles:
        return [box]

    left_indices = indices[sorted_idx[:mid_idx]]
    right_indices = indices[sorted_idx[mid_idx:]]

    # 재귀 호출
    return (get_balanced_overlapping_grids(particle_pos, left_indices, max_particles, min_particles, overlap_ratio, particle_mask) + 
            get_balanced_overlapping_grids(particle_pos, right_indices, max_particles, min_particles, overlap_ratio, particle_mask))

def tile_ranges(xmin, xmax, grid_size, stride):
    starts = []
    x = xmin
    while x < xmax:
        starts.append(x)
        x = x + grid_size
    
    ranges = []
    for i, s in enumerate(starts):
        e = s + grid_size * stride
        e = min(e, xmax)
        ranges.append((s,e))

    return ranges

if train_flag:
    os.mkdir(saving_path)
    os.mkdir(test_result_path)
    utils.save_code(saving_path)

    with open(saving_path + 'log_' + time + '.txt', 'w') as log_file:
        log_file.write("DEM simulation" + '\n')
        log_file.write('Training Start : ' + time + '\n')

node_input_size = network_attributes_pack.node_encoder.input_size
edge_input_size = network_attributes_pack.edge_encoder.input_size
target_input_size = network_attributes_pack.edge_decoder_pp.output_size

node_normalizer = normalizer.online_normalizer('node_normalizer', node_input_size, norm_acc_length, 1e-6, saving_path, 'cpu')
edge_normalizer = normalizer.online_normalizer('mesh_normalizer', edge_input_size, norm_acc_length, 1e-6, saving_path, 'cpu')
target_normalizer = normalizer.online_normalizer('target_normalizer', target_input_size, norm_acc_length, 1e-6, saving_path, 'cpu')

normalizer_pack = [node_normalizer, edge_normalizer, target_normalizer]

import torch
from dataset import DataPack, NodePack, EdgePack, TargetPack

def gns_collate_fn(batch):
    # 1. 노드 정보 합치기
    node_features = torch.cat([d.nodepack.node_features for d in batch], dim=0)
    
    # 2. 엣지 및 모든 마스크/벡터 정보 합치기 준비
    all_receivers, all_senders, all_edge_features = [], [], []
    all_pairwise_masks, all_reverse_idx = [], []
    all_edge_a, all_edge_b, all_edge_c = [], [], []
    all_b_deg, all_c_deg = [], []
    
    node_offset = 0
    edge_offset = 0  # 👈 reverse_edge_idx를 위한 엣지 오프셋!

    for d in batch:
        ep = d.edgepack
        
        # 노드 인덱스 시프트
        all_receivers.append(ep.receivers + node_offset)
        all_senders.append(ep.senders + node_offset)
        
        # 그냥 합치면 되는 텐서들
        all_edge_features.append(ep.edge_features)
        all_pairwise_masks.append(ep.pairwise_mask)
        all_edge_a.append(ep.edge_a)
        all_edge_b.append(ep.edge_b)
        all_edge_c.append(ep.edge_c)
        all_b_deg.append(ep.b_degenerate_mask)
        all_c_deg.append(ep.c_degenerate_mask)
        
        # 🚨 [매우 중요] 반대 방향 엣지 인덱스(reverse_edge_idx) 시프트 로직
        # -1은 반대 엣지가 없다는 뜻이므로, -1이 아닌 진짜 인덱스에만 누적된 엣지 개수를 더해줍니다.
        if ep.reverse_edge_idx is not None:
            rev_idx = ep.reverse_edge_idx.clone()
            valid_mask = rev_idx != -1
            rev_idx[valid_mask] += edge_offset
            all_reverse_idx.append(rev_idx)
            
        # 다음 배치를 위해 오프셋 증가
        node_offset += d.nodepack.node_features.size(0)
        edge_offset += ep.receivers.size(0)

    # 텐서 병합 (dim=0으로 길게 이어 붙이기)
    receivers = torch.cat(all_receivers, dim=0)
    senders = torch.cat(all_senders, dim=0)
    edge_features = torch.cat(all_edge_features, dim=0)
    pairwise_mask = torch.cat(all_pairwise_masks, dim=0)
    edge_a = torch.cat(all_edge_a, dim=0)
    edge_b = torch.cat(all_edge_b, dim=0)
    edge_c = torch.cat(all_edge_c, dim=0)
    b_degenerate_mask = torch.cat(all_b_deg, dim=0)
    c_degenerate_mask = torch.cat(all_c_deg, dim=0)
    reverse_edge_idx = torch.cat(all_reverse_idx, dim=0) if all_reverse_idx else None

    # 3. 타겟 정보 합치기
    normalized_target = torch.cat([d.targetpack.normalized_target for d in batch], dim=0)

    # 4. 완벽하게 조립된 DataPack 생성
    new_nodepack = batch[0].nodepack._replace(node_features=node_features)
    
    new_edgepack = batch[0].edgepack._replace(
        edge_features=edge_features,
        receivers=receivers,
        senders=senders,
        pairwise_mask=pairwise_mask,
        edge_a=edge_a, 
        edge_b=edge_b, 
        edge_c=edge_c,
        reverse_edge_idx=reverse_edge_idx,
        b_degenerate_mask=b_degenerate_mask,
        c_degenerate_mask=c_degenerate_mask
    )
    
    new_targetpack = batch[0].targetpack._replace(normalized_target=normalized_target)
    
    return DataPack(new_nodepack, new_edgepack, new_targetpack)

def pre_accumulation(i, data_set : dataset.gns_dataset):
    if i % 10 == 0:
        print(i, "step")
    # ✅ DataLoader를 쓰기 위해 get_data 대신 직접 index로 접근합니다.
    _ = data_set[i]

def collate_fn(batch):
    # batch는 리스트이고 각 요소는 get_data()의 결과 (namedtuple DataPack)
    # batch_size > 1일 때 리스트를 그대로 유지하여 Graph.forward가 배치 처리하도록 함
    return batch

def train_cycle(data_set : dataset.gns_dataset, test_set : dataset.gns_dataset):

    batch_size = 4

    steps_per_epoch = data_set.__len__() // batch_size
    train_length = steps_per_epoch * nepochs
    adjusted_lr_decay_length = lr_decay_length // batch_size

    lr_decay_calculator = utils.lr_decay_calculator(
        steps_per_epoch, 
        train_length, 
        adjusted_lr_decay_length, 
        decay_offset, 
        secondary_decay_offset
    )
    
    # dataset_length = data_set.__len__()
    # train_length = dataset_length * nepochs

    # lr_decay_calculator = utils.lr_decay_calculator(dataset_length, train_length, lr_decay_length, decay_offset, secondary_decay_offset)

    graph = gm.Graph(network_attributes_pack,
                     training_parameters_pack,
                     device,
                     train_flag)
    if not fresh_start:
        graph.load_network(test_network_path)
        graph.graph_net.train()
        for norm in normalizer_pack:
            norm.load_normalizer(test_network_path)

        
    
    if fresh_start:

        print("Start pre-accumulation")
        for i in range(pre_accumulation_steps):
            with torch.no_grad():
                pre_accumulation(i, data_set)
        alpha = (16 ** 0.5)

        for m in graph.graph_net.sub_nets.node_encoder.modules():
            if isinstance(m, torch.nn.Linear):
                with torch.no_grad():
                    m.weight[:,19:] *= alpha
                break

    t0 = datetime.now()
    loss_list = []

    batch_size = 4  # 배치 사이즈 증가로 GPU 활용도 높임
    num_workers = 8  # 데이터 로딩/전처리를 비동기로 수행
    
    train_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=gns_collate_fn,
    )

    step = 0

    for epoch in range(nepochs):

        print("Start epoch")

        for i, data_pack in enumerate(train_loader):

            if step % test_interval == 0 and epoch % 100 == 0:
                with torch.no_grad():
                    test_cycle(test_set, graph)

            graph.zero_grad()

            lr_decay = lr_decay_calculator.get_lrd(step % steps_per_epoch, epoch)

            graph.set_lr(lr * lr_decay)

            # 배치 처리 (len(data_pack) = batch_size)
            _, loss = graph.forward(data_pack)

            loss_list.append(loss[0])

            lr_decay_calculator.update_iterator()

            graph.train_step(grad_limit)

            step += 1

            if step % 10 == 0:
                log_text = ("Epochs : " + str(epoch) + 
                            " - Step : " + str(step) + 
                            " - Loss : " + str('%.8f' % loss[0]) + 
                            " - Abs Loss : " + str('%.8f' % loss[1]) + 
                            " - Lr : " + str('%.8f' % graph.graph_optimizer.param_groups[0]['lr']))
                print(log_text)

                if (step % 100 == 0):
                    with open(saving_path + 'log_' + time + '.txt', 'a') as log_file:
                        log_file.write(log_text + '\n')
                        log_file.write('Time : ' + str(datetime.now()) + '\n')

                if step % monitor_interval == 0:
                    torch.save(graph.graph_net, saving_path + 'graph_network.pt')

                    for norm in normalizer_pack:
                        norm.save_variables()

                    print("/// time passed since training start: ", datetime.now() - t0, " ///")

    plt.rcParams['figure.figsize'] = [20, 5]
    plt.plot(loss_list, label = ("loss",'loss_abs'))
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0.0, 0.1)
    plt.savefig(saving_path+'loss.png', dpi=200)
    plt.show()
    plt.close()
    plt.plot(loss_list, label = ("loss",'loss_abs'))
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0.0, 0.01)
    plt.savefig(saving_path+'loss_0_1.png', dpi=200)
    plt.show()
    plt.close()

    with torch.no_grad():
        test_cycle(test_set, graph, roll_out_flag=False, one_step_flag=True)

    if step % 10 == 0:
        log_text = ("Epochs : " + str(epoch) + 
                    " - Step : " + str(step) + 
                    " - Loss : " + str('%.8f' % loss[0]) + 
                    " - Abs Loss : " + str('%.8f' % loss[1]) + 
                    " - Lr : " + str('%.8f' % graph.graph_optimizer.param_groups[0]['lr']))
        print(log_text)

    if (step % 100 == 0):
        with open(saving_path + 'log_' + time + '.txt', 'a') as log_file:
            log_file.write(log_text + '\n')
            log_file.write('Time : ' + str(datetime.now()) + '\n')

    if step % monitor_interval == 0:
        torch.save(graph.graph_net, saving_path + 'graph_network.pt')

        for norm in normalizer_pack:
            norm.save_variables()

        print("/// time passed since training start: ", datetime.now() - t0, " ///")

    plt.rcParams['figure.figsize'] = [20, 5]
    plt.plot(loss_list, label = ("loss",'loss_abs'))
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0.0, 0.1)
    plt.savefig(saving_path+'loss.png', dpi=200)
    plt.show()
    plt.close()
    plt.plot(loss_list, label = ("loss",'loss_abs'))
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0.0, 0.01)
    plt.savefig(saving_path+'loss_0_1.png', dpi=200)
    plt.show()
    plt.close()

    with torch.no_grad():
        test_cycle(test_set, graph, roll_out_flag=False, one_step_flag=True)

def test_cycle(test_set : dataset.gns_dataset, graph : gm.Graph, plot_flag = False, cur_test_seqeunce_idx = test_sequence_idx, roll_out_flag = roll_out_flag, one_step_flag = one_step_flag):

    global test_length
    global test_loss_list

    pred_pos_list = []
    pred_vel_list = []

    for norm in normalizer_pack:
        norm.freeze = True

    if plot_flag:
        space = 60
        height, width = 600, 600

        empty_space = np.ones((height, space, 3), dtype=np.uint8) * 255

        fcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(test_network_path + 'zplane_large' + str(test_sequence_idx) + '.avi', fcc, 30.0, (1260, 600))

        if not out.isOpened():
            print("Video Writer open failed with DIVX")
            fcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(test_network_path + 'zplane' + str(test_sequence_idx) + '.avi', fcc, 30.0, (1260,600))

            assert out.isOpened(), "Video Writer failed to open with both DIVX and MJPG"

    t0 = datetime.now()

    sequence_length = test_set.__len__()

    test_set.noise_level = 0.0

    raw_data_container = [test_set.get_raw_data(i) for i in range(sequence_length)]

    target_sequence = [test_set.return_target_sequence(raw_data)[None,:,:].detach().cpu() for raw_data in raw_data_container]

    updated_prev_pos = None
    updated_pos = None
    updated_vel = None
    updated_acc = None

    t1 = datetime.now()

    if not roll_out_flag:
        test_length = len(raw_data_container)

    graph.train_flag = False
    prediction_sequence = [target_sequence[0]]

    i = 0
    stop_flag = False

    loss_list = []
    loss_abs_list = []

    pred_particle_id = raw_data_container[0].particle_id.clone().cpu()
    if pred_particle_id.ndim == 2 and pred_particle_id.shape[-1] == 1:
        pred_particle_id = pred_particle_id.squeeze(-1)
    pred_particle_id = pred_particle_id.bool()

    while not stop_flag:
        
        if i < len(raw_data_container):
            raw_data_pack = raw_data_container[i]
        else:
            raw_data_pack = raw_data_container[-1]

        #raw_data_pack.todevice(test_set.device)

        if one_step_flag == True or updated_pos == None:
            updated_vel, updated_prev_pos, updated_pos, updated_acc = test_set.data_from_test_set(raw_data_pack)

        raw_data_pack = test_set.update_raw_data(copy.deepcopy(raw_data_pack), updated_vel, updated_prev_pos, updated_pos, updated_acc)

        if one_step_flag == False:
            cur_particle_pos = raw_data_pack.particle_pos  # (N_p, 3)
            px = cur_particle_pos[:, 0]
            py = cur_particle_pos[:, 1]
            pz = cur_particle_pos[:, 2]

            spatial_mask = (px >= -60) & (px <= 60) & (py >= -40) & (py <= 60) & (pz >= -10) & (pz <= 100)  # (N_p,)

            #pred_particle_id = spatial_mask & pred_particle_id # (N_p,)
            spatial_mask = spatial_mask.cpu()
            pred_particle_id = pred_particle_id.cpu()
            pred_particle_id = spatial_mask & pred_particle_id # (N_p,)

            # time 축 전체에 대해 동일한 particle index만 유지
            particle_id_roll = torch.zeros_like(raw_data_pack.particle_id)

            particle_id_roll[pred_particle_id] = 1

            raw_data_pack = dc_replace(raw_data_pack, particle_id = particle_id_roll)

        data_pack = test_set.update_data(raw_data_pack, contact_distance)

        network_output, loss = graph.forward(data_pack)

        pos_prediction, vel_prediction, acc_prediction = test_set.reverse_output(network_output[:updated_pos.shape[0]], updated_pos, updated_vel)

        pred_pos_list.append(pos_prediction.detach().cpu())
        pred_vel_list.append(vel_prediction.detach().cpu())

        if plot_flag:

            mesh_node_type = raw_data_pack.mesh_node_type
            hopper_indices = torch.where(mesh_node_type == 1, 1, 0).nonzero().squeeze()
            roll1_indices= torch.where(mesh_node_type == 2, 1, 0).nonzero().squeeze()
            roll2_indices = torch.where(mesh_node_type == 3, 1, 0).nonzero().squeeze()

            particle_prediction = pos_prediction[data_pack.nodepack.particle_indices.detach().cpu().numpy()].detach().cpu().numpy()
            hopper = raw_data_pack.next_mesh_pos[hopper_indices.detach().cpu().numpy()].detach().cpu().numpy()
            roller1 = raw_data_pack.next_mesh_pos[roll1_indices.detach().cpu().numpy()].detach().cpu().numpy()
            roller2 = raw_data_pack.next_mesh_pos[roll2_indices.detach().cpu().numpy()].detach().cpu().numpy()

            element = raw_data_pack.cells.detach().cpu().numpy()

            equipment = np.vstack([hopper, roller1, roller2])

            particle_target = raw_data_pack.next_particle_pos[data_pack.nodepack.next_particle_indices.detach().cpu().numpy()].detach().cpu().numpy()

            fig = plt.figure(figsize=(6,6), facecolor='white')
            fig2 = plt.figure(figsize=(6,6), facecolor='white')

            ax = fig.add_subplot(111, projection='3d')
            ax2 = fig2.add_subplot(111, projection='3d')

            ax.scatter(particle_prediction[:,0], particle_prediction[:,1], particle_prediction[:,2], s = 5, c = 'green', label = 'particle')
            ax.plot_trisurf(equipment[:,0], equipment[:,1], equipment[:,2], triangles = element[:,1:], color = 'red', label = 'equipment', alpha = 0.3)
            ax.set_xlim([-25, 25])
            ax.set_ylim([-25, 25])
            ax.set_zlim([-0.02, 40])
            ax.set_box_aspect((50,50,40.02))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_axis_off()
            ax.set_title('prediction')
            ax.legend(loc = 'upper right')
            ax.view_init(elev=20, azim=45, vertical_axis='y')
            # ax.view_init(0,0, roll=90)
            # ax.view_init(90,-90)

            ax2.scatter(particle_target[:,0], particle_target[:,1], particle_target[:,2], s = 5, c = 'green', label = 'particle')
            ax2.plot_trisurf(equipment[:,0], equipment[:,1], equipment[:,2], triangles = element[:,1:], color = 'red', label = 'equipment', alpha = 0.3)
            ax2.set_xlim([-25, 25])
            ax2.set_ylim([-25, 25])
            ax2.set_zlim([-0.02, 25])
            ax2.set_box_aspect((50,50,40.02))
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_zticks([])
            ax2.set_axis_off()
            ax2.set_title('target')
            ax2.legend(loc = 'upper right')
            ax2.view_init(elev=20, azim=45, vertical_axis='y')
            # ax2.view_init(0,0,roll=90)
            # ax2.view_init(90,-90)

            fig.canvas.draw()
            fig2.canvas.draw()

            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
            image2 = np.frombuffer(fig2.canvas.buffer_rgba(), dtype=np.uint8)
            image2 = image2.reshape(fig2.canvas.get_width_height()[::-1] + (4,))[..., :3]

            combined_image = np.hstack((image, empty_space, image2))

            out.write(combined_image)

            plt.close(fig)
            plt.close(fig2)


        loss_list.append(loss[0])
        loss_abs_list.append(loss[1])

        log_text = ("Epochs : " + str(i) +
                    " - Loss : " + str('%.8f' % loss[0]) + 
                    " - Abs Loss : " + str('%.8f' % loss[1]))
        print(log_text)

        updated_prev_pos, updated_pos, updated_vel, updated_acc = test_set.update_test_data(raw_data_pack, updated_prev_pos, updated_pos, updated_vel, updated_acc, pos_prediction, vel_prediction, acc_prediction)

        prediction_sequence.append(torch.cat((updated_pos.detach().cpu(), updated_vel[:,-3:].detach().cpu(), updated_acc.detach().cpu(), raw_data_pack.particle_id.unsqueeze(-1).cpu()), dim = 1)[None, :, :])

        print("Test Cycle : ", datetime.now() - t1)

        if i == 10: # 디버깅 시 숫자, 본 학습 시 test_length
            stop_flag = True
        i += 1
        
    if plot_flag:

        out.release()
        cv2.destroyAllWindows()

    t2 = datetime.now()
    graph.train_flag = True
    
    prediction_sequence = torch.cat(prediction_sequence, dim=0).detach().cpu().numpy()
    target_sequence = torch.cat(target_sequence, dim=0).detach().cpu().numpy()

    if train_flag:
    
        np.save((test_result_path + "/pred.npy"), prediction_sequence)
        np.save((test_result_path + "/targ.npy"), target_sequence)
        # np.save((test_result_path + "/node_type.npy"), raw_data_container[0].node_type.detach().cpu().numpy())
    else:
        np.save((test_network_path + "/pred_width_25.npy"), prediction_sequence)
        np.save((test_network_path + "/targ_width_25.npy"), target_sequence)
    
    for norm in normalizer_pack:
        norm.freeze = False
        
    loss_list = torch.tensor(loss_list, dtype = torch.float32, device=device, requires_grad=False)
    loss_abs_list = torch.tensor(loss_abs_list, dtype = torch.float32, device=device, requires_grad=False)
    print("Loss mean: " + str("%.8f" % (torch.mean(loss_list).item()))
          + " - Loss std : " + str("%.8f" % (torch.std(loss_list).item()))
          + " - Loss max : " + str("%.8f" % (torch.max(loss_list).item()))
          + " - RMSE Loss : " + str('%.8f' % (torch.mean(loss_abs_list).item()))
          + " - RMSE Loss std : " + str('%.8f' % (torch.std(loss_abs_list).item()))
          + " - RMSE Loss max : " + str('%.8f' % (torch.max(loss_abs_list).item())))
    print("Test run without plotting: ", t2 - t0)        
    print("whole test run: ", datetime.now() - t0)
    print("current time: ", datetime.now())

    test_loss_list.append([cur_test_seqeunce_idx, torch.mean(loss_abs_list).item(), torch.max(loss_abs_list).item()])
    print([cur_test_seqeunce_idx, torch.mean(loss_abs_list).item(), torch.max(loss_abs_list).item()])

    print("Saving rollout predictions...")
    pred_pos_tensor = torch.stack(pred_pos_list)
    pred_vel_tensor = torch.stack(pred_vel_list)
    
    save_dict = {
        'anti_pos': pred_pos_tensor,
        'anti_vel': pred_vel_tensor
    }
    
    # 에폭 번호를 알 수 있다면 파일명에 포함시키는 것이 좋습니다.
    save_path = 'rollout_predictions.pt' 
    torch.save(save_dict, save_path)
    print(f"✅ Successfully saved to {save_path}")

def grid_test_cycle(test_set, graph, plot_flag, test_sequence_idx, max_particles, min_particles, overlap_ratio):

    global test_length
    global test_loss_list

    merge_mode = 'avg'

    for norm in normalizer_pack:
        norm.freeze = True

    t0 = datetime.now()

    sequence_length = test_set.__len__()

    test_set.noise_level = 0.0

    raw_data_container = [test_set.get_raw_data(i) for i in range(sequence_length)]

    target_sequence = [test_set.return_target_sequence(raw_data)[None,:,:].detach().cpu() for raw_data in raw_data_container]

    updated_prev_pos = None
    updated_pos = None
    updated_vel = None
    updated_acc = None

    if plot_flag:
        space = 60
        height, width = 600, 600

        empty_space = np.ones((height, space, 3), dtype=np.uint8) * 255
        fcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(test_network_path + 'xplane_large' + str(test_sequence_idx) + '.mp4', fcc, 30.0, (1260, 600))
        # fcc = cv2.VideoWriter_fourcc(*'DIVX')
        # out = cv2.VideoWriter(test_network_path + 'xplane_large' + str(test_sequence_idx) + '.avi', fcc, 30.0, (1260, 600))
                    
    t1 = datetime.now()

    if not roll_out_flag:
        test_length = len(raw_data_container)

    graph.train_flag = False
    prediction_sequence = [target_sequence[0]]

    i = 0
    stop_flag = False

    pred_particle_id = raw_data_container[0].particle_id.clone()
    if pred_particle_id.ndim == 2 and pred_particle_id.shape[-1] == 1:
        pred_particle_id = pred_particle_id.squeeze(-1)
    pred_particle_id = pred_particle_id.bool()

    while not stop_flag:
        test_set.maximum_mesh_edges = 0
        test_set.maximum_particle_edges = 0
        test_set.maximum_particle_edges_node = 0
        test_set.maximum_mesh_edges_node = 0
        torch.cuda.reset_peak_memory_stats()
        mem(f"step {i} before")
        if i < len(raw_data_container):
            raw_data_pack = raw_data_container[i]
        else:
            raw_data_pack = raw_data_container[-1]

        if one_step_flag == True or updated_pos == None:
            updated_vel, updated_prev_pos, updated_pos, updated_acc = test_set.data_from_test_set(raw_data_pack)

        raw_data_pack = test_set.update_raw_data(copy.deepcopy(raw_data_pack), updated_vel, updated_prev_pos, updated_pos, updated_acc)

        if one_step_flag == False:
            cur_particle_pos = raw_data_pack.particle_pos  # (N_p, 3)
            px = cur_particle_pos[:, 0]
            py = cur_particle_pos[:, 1]
            pz = cur_particle_pos[:, 2]

            spatial_mask = (px >= X_BOUND[0]*1.05) & (px <= X_BOUND[1]*1.05) & (py >= Y_BOUND[0]*1.05) & (py <= Y_BOUND[1]*1.05) & (pz >= Z_BOUND[0]*1.05) & (pz <= Z_BOUND[1]*1.05)  # (N_p,)

            pred_particle_id = spatial_mask & pred_particle_id # (N_p,)

            # time 축 전체에 대해 동일한 particle index만 유지
            particle_id_roll = torch.zeros_like(raw_data_pack.particle_id)

            particle_id_roll[pred_particle_id] = 1

            raw_data_pack = dc_replace(raw_data_pack, particle_id = particle_id_roll)

        all_indices = pred_particle_id.nonzero(as_tuple=False).reshape(-1)

        adaptive_boxes = get_balanced_overlapping_grids(raw_data_pack.particle_pos,
                                                        all_indices,
                                                        max_particles,
                                                        min_particles,
                                                        overlap_ratio,
                                                        raw_data_pack.particle_id)

        # adaptive_boxes = ((-26,26,-40,23,-10,10),(-26,26,-40,23,0,20),(-26,26,-40,23,10,30),(-26,26,-40,23,20,40),(-26,26,-40,23,30,50),(-26,26,-40,23,40,60),(-26,26,-40,23,50,70),(-26,26,-40,23,60,80),(-26,26,-40,23,70,90),(-26,26,-40,23,80,100),(-26,26,-40,23,90,110))

        entire_pos_prediction = raw_data_pack.particle_pos.clone()
        entire_vel_prediction = raw_data_pack.particle_vel.clone()
        entire_acc_prediction = raw_data_pack.acc.clone()

        pos_sum = torch.zeros_like(raw_data_pack.particle_pos)
        vel_sum = torch.zeros_like(raw_data_pack.particle_vel)
        acc_sum = torch.zeros_like(raw_data_pack.acc)
        count = torch.zeros_like(raw_data_pack.particle_pos[:,0], dtype=torch.float32)

        max_num_particles = 0
        max_num_edges = 0
        min_num_particles = 1e9

        print(f"Step {i}: Grids={len(adaptive_boxes)} (Overlapping)")

        for (xl, xr, yl, yr, zl, zr) in adaptive_boxes:
            sliced_raw_data_pack, particle_idx = test_set.build_tiled_raw_data(raw_data_pack, (xl, xr), (yl, yr), (zl, zr), contact_distance)

            if sliced_raw_data_pack.particle_pos.shape[0] == 0:
                continue

            data_pack = test_set.update_data(sliced_raw_data_pack, contact_distance)

            network_output = graph.forward(data_pack, train_flag = False, grid_flag = True)

            pos_prediction, vel_prediction, acc_prediction = test_set.reverse_output(network_output[:sliced_raw_data_pack.particle_pos.shape[0]], 
                                                                                    sliced_raw_data_pack.particle_pos,
                                                                                    sliced_raw_data_pack.particle_vel)
            
            center_x = (xl + xr) / 2
            center_y = (yl + yr) / 2
            center_z = (zl + zr) / 2

            radius_x = (xr - xl) / 2.0
            radius_y = (yr - yl) / 2.0
            radius_z = (zr - zl) / 2.0

            sigma = min(radius_x, radius_y, radius_z) / 2.0

            px, py, pz = sliced_raw_data_pack.particle_pos[:,0], sliced_raw_data_pack.particle_pos[:,1], sliced_raw_data_pack.particle_pos[:,2]

            dist_x = torch.min(px - xl, xr - px)
            dist_y = torch.min(py - yl, yr - py)
            dist_z = torch.min(pz - zl, zr - pz)

            min_dist = torch.min(torch.stack([dist_x, dist_y, dist_z], dim = 1), dim = 1)[0]
            min_dist = torch.clamp(min_dist, min=0.0)

            blend_width = 4.0

            weight = torch.clamp(min_dist / blend_width, 0.0, 1.0).unsqueeze(-1)

            # dist_sq = (px-center_x)**2 + (py-center_y)**2 + (pz-center_z)**2
            # weight = torch.exp(-dist_sq / (2 * sigma ** 2)).unsqueeze(-1)
            
            if merge_mode == 'avg':
                pos_sum[particle_idx] += (pos_prediction[:sliced_raw_data_pack.particle_pos.shape[0]] * weight).detach().cpu()
                vel_sum[particle_idx] += (vel_prediction[:sliced_raw_data_pack.particle_pos.shape[0]] * weight).detach().cpu()
                acc_sum[particle_idx] += (acc_prediction[:sliced_raw_data_pack.particle_pos.shape[0]] * weight).detach().cpu()
                count[particle_idx] += weight.squeeze(-1).detach().cpu()

            if max_num_particles < data_pack.nodepack.particle_indices.shape[0]:
                max_num_particles = data_pack.nodepack.particle_indices.shape[0]

            if min_num_particles > data_pack.nodepack.particle_indices.shape[0]:
                min_num_particles = data_pack.nodepack.particle_indices.shape[0]

            if max_num_edges < data_pack.edgepack.edge_features.shape[0]:
                max_num_edges = data_pack.edgepack.edge_features.shape[0]
                max_box = [[xl, xr], [yl, yr], [zl, zr]]

        if merge_mode == 'avg':

            valid = count > 0
            entire_pos_prediction[valid] = pos_sum[valid] / count[valid].unsqueeze(1)
            entire_vel_prediction[valid] = vel_sum[valid] / count[valid].unsqueeze(1)
            entire_acc_prediction[valid] = acc_sum[valid] / count[valid].unsqueeze(1)
            

        if plot_flag:

            mesh_node_type = raw_data_pack.mesh_node_type
            hopper_indices = torch.where(mesh_node_type == 1, 1, 0).nonzero().squeeze()
            roll1_indices= torch.where(mesh_node_type == 2, 1, 0).nonzero().squeeze()
            roll2_indices = torch.where(mesh_node_type == 3, 1, 0).nonzero().squeeze()

            next_particle_indices = raw_data_pack.next_particle_id.nonzero(as_tuple=False).reshape(-1)
            particle_indices = raw_data_pack.particle_id.nonzero(as_tuple=False).reshape(-1)

            particle_prediction = entire_pos_prediction[particle_indices.detach().cpu().numpy()].detach().cpu().numpy()
            hopper = raw_data_pack.next_mesh_pos[hopper_indices.detach().cpu().numpy()].detach().cpu().numpy()
            roller1 = raw_data_pack.next_mesh_pos[roll1_indices.detach().cpu().numpy()].detach().cpu().numpy()
            roller2 = raw_data_pack.next_mesh_pos[roll2_indices.detach().cpu().numpy()].detach().cpu().numpy()

            element = raw_data_pack.cells.detach().cpu().numpy()

            equipment = np.vstack([hopper, roller1, roller2])

            particle_target = raw_data_pack.next_particle_pos[next_particle_indices.detach().cpu().numpy()].detach().cpu().numpy()

            fig = plt.figure(figsize=(6,6), facecolor='white')
            fig2 = plt.figure(figsize=(6,6), facecolor='white')

            ax = fig.add_subplot(111, projection='3d')
            ax2 = fig2.add_subplot(111, projection='3d')

            ax.scatter(particle_prediction[:,0], particle_prediction[:,1], particle_prediction[:,2], s = 5, c = 'green', label = 'particle')
            ax.plot_trisurf(equipment[:,0], equipment[:,1], equipment[:,2], triangles = element[:,1:], color = 'red', label = 'equipment', alpha = 0.3)
            ax.set_xlim(X_BOUND)
            ax.set_ylim(Y_BOUND)
            ax.set_zlim(Z_BOUND)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_axis_off()
            ax.set_title('prediction')
            ax.legend(loc = 'upper right')
            # ax.view_init(elev=45, azim=-45, roll=0)
            # ax.view_init(0,0, roll=90)
            # ax.view_init(90,-90)
            ax.view_init(elev=20, azim=45, vertical_axis='y')

            ax2.scatter(particle_target[:,0], particle_target[:,1], particle_target[:,2], s = 5, c = 'green', label = 'particle')
            ax2.plot_trisurf(equipment[:,0], equipment[:,1], equipment[:,2], triangles = element[:,1:], color = 'red', label = 'equipment', alpha = 0.3)
            ax2.set_xlim(X_BOUND)
            ax2.set_ylim(Y_BOUND)
            ax2.set_zlim(Z_BOUND)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_zticks([])
            ax2.set_axis_off()
            ax2.set_title('target')
            ax2.legend(loc = 'upper right')
            # ax2.view_init(elev=45, azim=-45, roll=0)
            # ax2.view_init(0,0,roll=90)
            # ax2.view_init(90,-90)
            ax2.view_init(elev=20, azim=45, vertical_axis='y')

            fig.canvas.draw()
            fig2.canvas.draw()

            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
            image2 = np.frombuffer(fig2.canvas.buffer_rgba(), dtype=np.uint8)
            image2 = image2.reshape(fig2.canvas.get_width_height()[::-1] + (4,))[..., :3]

            combined_image = np.hstack((image, empty_space, image2))

            out.write(combined_image)

            plt.close(fig)
            plt.close(fig2)

        # maximum_edge_image(raw_data_pack, max_box[0], max_box[1], max_box[2], test_set, graph, f'Epoch{i}_x{max_box[0][0]}_{max_box[0][1]}_y{max_box[1][0]}_{max_box[1][1]}_z{max_box[2][0]}_{max_box[2][1]}.png')

        updated_prev_pos, updated_pos, updated_vel, updated_acc = test_set.update_test_data(raw_data_pack, 
                                                                                             updated_prev_pos, updated_pos, 
                                                                                             updated_vel, 
                                                                                             updated_acc, 
                                                                                             entire_pos_prediction, entire_vel_prediction, entire_acc_prediction)
        
        prediction_sequence.append(torch.cat((updated_pos, updated_vel[:,-3:], updated_acc, raw_data_pack.particle_id.unsqueeze(-1).to(updated_pos.device)), dim = 1)[None, :, :])

        print("maximum number of particles : ", max_num_particles)
        print("minimum number of particlces : ", min_num_particles)
        print("maximum number of edges : ", max_num_edges)
        print("Num particles : ", test_set.maximum_mesh_edges_node, "maximum number of mesh edges : ", test_set.maximum_mesh_edges)
        print("Num particles : ", test_set.maximum_particle_edges_node, "maximum number of particle edges : ", test_set.maximum_particle_edges)
        print("Epoch : ", i)
        print("Test Cycle : ", datetime.now() - t1)


        if i == test_length:
            stop_flag = True
        mem(f"step {i} after")
        i += 1

    t2 = datetime.now()
    graph.train_flag = True
    
    prediction_sequence = torch.cat(prediction_sequence, dim=0).detach().cpu().numpy()
    target_sequence = torch.cat(target_sequence, dim=0).detach().cpu().numpy()
    if plot_flag:

        out.release()
        cv2.destroyAllWindows()

    if train_flag:
    
        np.save((test_result_path + "/pred.npy"), prediction_sequence)
        np.save((test_result_path + "/targ.npy"), target_sequence)
        # np.save((test_result_path + "/node_type.npy"), raw_data_container[0].node_type.detach().cpu().numpy())
    else:
        np.save((test_network_path + "/pred.npy"), prediction_sequence)
        np.save((test_network_path + '/targ.npy'), target_sequence)
    
    for norm in normalizer_pack:
        norm.freeze = False
        
    print("Test run without plotting: ", t2 - t0)        
    print("whole test run: ", datetime.now() - t0)
    print("current time: ", datetime.now())

def main():

    if train_flag:
        test_set = dataset.gns_dataset(data_parameters_pack, normalizer_pack, device, mode='test')
        test_set.load_dataset(test_set.ds_path + test_set.test_folder)

        data_set = dataset.gns_dataset(data_parameters_pack, normalizer_pack, device, mode='train')
        data_set.load_dataset(data_set.ds_path + data_set.training_folder)

        test_set.noise_level = 0.0
        test_set.set_sequence(test_sequence_idx)

        train_cycle(data_set, test_set)

    else:
        test_set = dataset.gns_dataset(data_parameters_pack, normalizer_pack, device)

        test_sequence_list = list(range(100))

        graph = gm.Graph(network_attributes_pack, training_parameters_pack, device, train_flag = False)
        graph.load_network(test_network_path)

        for norm in normalizer_pack:
            norm.load_normalizer(test_network_path)

        # for test_sequence_idx in test_sequence_list:

        #     print("test seqeunce idx : " + str(test_sequence_idx))
            
        #     test_set.load_dataset(test_set.ds_path + test_set.test_folder)
        #     test_set.noise_level = 0.0
        #     test_set.set_sequence(test_sequence_idx)

        #     with torch.no_grad():

        #         test_cycle(test_set, graph, plot_flag, test_sequence_idx)

        # test_loss_list.sort(key = lambda x : x[1])

        # with open(test_network_path + 'test_loss.txt', 'a') as log_file:
        #     log_file.write('test_sequence_idx RMSE_mean RMSE_max' + '\n')        

        # for test_sequence_idx, loss_mean, loss_max in test_loss_list:
        #     loss_text = str(test_sequence_idx) + " " + str('%.8f' % loss_mean)  + " " + str('%.8f' % loss_max)
        #     with open(test_network_path + 'test_loss.txt', 'a') as log_file:
        #         log_file.write(loss_text + '\n')
        #     print(loss_text)


        test_set.load_dataset(test_set.ds_path + test_set.test_folder)
        test_set.noise_level = 0.0
        test_set.set_sequence(test_sequence_idx)
        with torch.no_grad():

            grid_test_cycle(test_set, graph, plot_flag, test_sequence_idx, 25000, 10000, 1)


if __name__ == '__main__':
    main()

