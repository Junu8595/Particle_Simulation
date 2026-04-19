import torch
import graph_networks as gns
from torch_scatter import scatter_softmax
import torch_scatter

class Graph():
    def __init__(self,
                 network_attribute_pack,
                 training_parameter_pack,
                 device,
                 train_flag):
        
        self.train_flag = train_flag
        self.device = device

        self.graph_net = gns.graph_net(network_attribute_pack).to(self.device)

        self.message_passing_steps = training_parameter_pack.message_passing_steps

        self.graph_optimizer = torch.optim.Adam(self.graph_net.parameters(), betas=(0.9,0.999), weight_decay=0.0)

        self.network_attribute_pack = network_attribute_pack

        self.encode = self.encoder
        self.process = self.processor
        self.decode = self.decoder

        self._debug_step_count = 0  # decoder 축별 디버그 로깅용 카운터


    def _move_datapack_to_device(self, datapack, device):
        """datapack 내부의 모든 텐서를 지정된 device로 이동"""
        def move_tensor_or_tuple(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, tuple):
                return tuple(move_tensor_or_tuple(item, device) for item in obj)
            else:
                return obj
        
        # nodepack 이동
        nodepack = datapack.nodepack
        moved_nodepack = tuple(move_tensor_or_tuple(field, device) for field in nodepack)
        
        # edgepack 이동
        edgepack = datapack.edgepack
        moved_edgepack = tuple(move_tensor_or_tuple(field, device) for field in edgepack)
        
        # targetpack 이동
        targetpack = datapack.targetpack
        moved_targetpack = tuple(move_tensor_or_tuple(field, device) for field in targetpack)
        
        from dataset import DataPack, NodePack, EdgePack, TargetPack
        
        # nodepack 재생성
        moved_nodepack = NodePack(*moved_nodepack)
        
        # edgepack 재생성
        moved_edgepack = EdgePack(*moved_edgepack)
        
        # targetpack 재생성
        moved_targetpack = TargetPack(*moved_targetpack)
        
        return DataPack(moved_nodepack, moved_edgepack, moved_targetpack)
    
    def forward_single(self, datapack, train_flag = True, grid_flag = False):
        """단일 그래프 샘플 처리"""
        # ✅ CPU에서 온 datapack을 GPU로 이동
        datapack = self._move_datapack_to_device(datapack, self.device)
        self.encode(datapack)
        self.process(self.message_passing_steps)
        return self.decode(datapack.targetpack, train_flag, grid_flag)
    
    def forward(self, datapack_or_list, train_flag = True, grid_flag = False):
        """단일 또는 배치 처리 지원"""
        # 배치인 경우 (리스트)
        if isinstance(datapack_or_list, list):
            loss_list = []
            for datapack in datapack_or_list:
                _, loss = self.forward_single(datapack, train_flag, grid_flag)
                loss_list.append(loss)
            
            # 배치 내 손실 평균
            avg_loss_0 = sum(l[0] for l in loss_list) / len(loss_list)
            avg_loss_1 = sum(l[1] for l in loss_list) / len(loss_list) if len(loss_list[0]) > 1 else 0
            return None, [avg_loss_0, avg_loss_1]
        else:
            # 단일 샘플
            return self.forward_single(datapack_or_list, train_flag, grid_flag)

    def encoder(self, datapack):
        self.next_particle_indices = datapack.nodepack.next_particle_indices
        self.receivers = datapack.edgepack.receivers
        self.senders = datapack.edgepack.senders

        node_features = datapack.nodepack.node_features
        edge_features = datapack.edgepack.edge_features
        
        # edge local frame 정보 저장
        self.edge_a = datapack.edgepack.edge_a
        self.edge_b = datapack.edgepack.edge_b
        self.edge_c = datapack.edgepack.edge_c
        self.reverse_edge_idx = datapack.edgepack.reverse_edge_idx
        self.pairwise_mask = datapack.edgepack.pairwise_mask
        self.b_degenerate_mask = datapack.edgepack.b_degenerate_mask
        self.c_degenerate_mask = datapack.edgepack.c_degenerate_mask
        
        self.latent_node = self.graph_net(self.graph_net.sub_nets.node_encoder, node_features)
        self.latent_edge = self.graph_net(self.graph_net.sub_nets.edge_encoder, edge_features)


    def processor(self, message_passing_step):
        for step in range(message_passing_step):
            self.latent_edge = self.update_edge(self.graph_net.sub_nets.edge_messenger,
                                                self.graph_net.sub_nets.edge_attention,
                                                self.latent_node, 
                                                self.latent_edge,
                                                self.receivers, 
                                                self.senders, 
                                                step)
            
            self.latent_node = self.update_node(self.graph_net.sub_nets.node_messenger, 
                                                self.latent_node, 
                                                self.latent_edge,
                                                self.receivers, 
                                                step)

    def decoder(self, targetpack, train_flag, grid_flag):
        # 1. 엣지 마스크 분리 (Particle-Particle vs Particle-Mesh)
        pw_mask = self.pairwise_mask # (E,) boolean mask
        pm_mask = ~pw_mask # 나머지는 모두 Mesh와의 Contact edge

        num_edges = self.latent_edge.shape[0]
        edge_scalars = torch.zeros((num_edges, 3), device=self.device, dtype=self.latent_edge.dtype)

        # 2. 각각의 디코더에 통과시켜 3개의 스칼라 계수(s1, s2, s3) 추출
        if pw_mask.any():
            edge_scalars[pw_mask] = self.graph_net(self.graph_net.sub_nets.edge_decoder_pp, self.latent_edge[pw_mask])
        
        if pm_mask.any():
            edge_scalars[pm_mask] = self.graph_net(self.graph_net.sub_nets.edge_decoder_pm, self.latent_edge[pm_mask])

        # 3. Vectorization: 스칼라 계수 * Edge Local Frame (a, b, c)
        # f_ij shape: (E, 3) - Global Frame에서의 3D 힘 벡터
        f_ij = (edge_scalars[:, 0:1] * self.edge_a + 
                edge_scalars[:, 1:2] * self.edge_b + 
                edge_scalars[:, 2:3] * self.edge_c)

        # 4. Aggregation: 엣지 벡터들을 수신 노드(receivers) 기준으로 합산
        num_nodes = self.latent_node.shape[0]
        edge_output = torch.zeros((num_nodes, 3), device=self.device, dtype=f_ij.dtype)
        edge_output = torch_scatter.scatter_add(f_ij, self.receivers, dim=0, out=edge_output)

        # 4b. Node-level residual decoder
        node_residual = self.graph_net(self.graph_net.sub_nets.node_decoder, self.latent_node)
        output = edge_output + node_residual

        if grid_flag:
            return output

        # 5. 축별 디버그 로깅 (train 시에만, 매 10000 decoder 호출마다)
        if train_flag:
            self._debug_step_count += 1
            if self._debug_step_count % 10000 == 0:
                with torch.no_grad():
                    def _log_axis(t, label):
                        print(f"[DEBUG step={self._debug_step_count}] {label}: "
                              f"mean=({t[:,0].mean():.5f}, {t[:,1].mean():.5f}, {t[:,2].mean():.5f})  "
                              f"std=({t[:,0].std():.5f}, {t[:,1].std():.5f}, {t[:,2].std():.5f})")
                    _log_axis(output[self.next_particle_indices], "output[particles] (X,Y,Z)")
                    _log_axis(node_residual[self.next_particle_indices], "node_residual[particles] (X,Y,Z)")
                    if pw_mask.any():
                        _log_axis(f_ij[pw_mask], "PP f_ij (X,Y,Z)")
                    if pm_mask.any():
                        _log_axis(f_ij[pm_mask], "PM f_ij (X,Y,Z)")

        # 6. Loss 계산 (Classic GNN 방식 — relative RMSE)
        error = targetpack.normalized_target - output

        self.loss = torch.pow(error[self.next_particle_indices], 2).sum(dim=1).mean()
        self.sum = torch.pow(targetpack.normalized_target[self.next_particle_indices], 2).sum(dim=1).mean()
        self.loss = torch.sqrt(self.loss) / self.sum

        loss_average = [self.loss.item(), error[self.next_particle_indices].abs().mean().item()]

        return output, loss_average



    def update_edge(self, edge_messenger, attention_messenger, latent_node_features, latent_edge_features, receiver_indices, sender_indices, step):
        
        edge_features = torch.hstack((latent_edge_features, latent_node_features[receiver_indices], latent_node_features[sender_indices]))
        attention = self.graph_net(attention_messenger, edge_features)
        attention = scatter_softmax(attention.squeeze(-1), index=receiver_indices)

        return attention.unsqueeze(-1) * self.graph_net(edge_messenger, edge_features, latent_edge_features, step)

    def update_node(self, node_messenger, node_features, edge_features, receivers, step):
        features = node_features
        edge_update_node = torch.scatter_add(torch.zeros(node_features.shape, device=node_features.device),
                                             0,
                                             receivers[:,None].expand(-1,node_features.shape[1]).to(node_features.device),
                                             edge_features.to(node_features.device))
        
        features = torch.hstack((features, edge_update_node))

        return self.graph_net(node_messenger, features, node_features, step)
    
    def train_step(self, threshold):

        if not torch.isfinite(self.loss):
            return
        
        self.loss.backward()

        torch.nn.utils.clip_grad_norm_(self.graph_net.parameters(), max_norm=threshold, norm_type=2)

        self.graph_optimizer.step()

    def zero_grad(self):
        self.graph_optimizer.zero_grad()
    
    def set_lr(self, lrd):
        for param in self.graph_optimizer.param_groups:
            param['lr'] = lrd

    def load_network(self, load_path):
        self.graph_net = self.graph_net.to(self.device)
 
        checkpoint = torch.load(load_path + self.graph_net.name + ".pt", map_location=self.device)
                
        self.graph_net.load_state_dict(checkpoint.state_dict())
        
        for i, sub_net in enumerate(self.graph_net.sub_nets):
            sub_net = sub_net.to(self.device)
    
        self.graph_net.eval()

        print("LOADING THE NETWORK IS FINISHED!")
