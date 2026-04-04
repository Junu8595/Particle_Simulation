import torch
import graph_networks as gns
from torch_scatter import scatter_softmax

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


    def forward(self,datapack, train_flag = True, grid_flag = False):
        self.encode(datapack)
        self.process(self.message_passing_steps)
        return self.decode(datapack.targetpack, train_flag, grid_flag)

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
        output = torch.zeros((num_nodes, 3), device=self.device, dtype=f_ij.dtype)
        
        # 여기서 scatter_add를 사용하기 위해 torch_scatter import 필요
        import torch_scatter
        output = torch_scatter.scatter_add(f_ij, self.receivers, dim=0, out=output)

        if grid_flag:
            return output

        # 5. Loss 계산 (기존 로직 유지)
        error = targetpack.normalized_target - output

        self.loss = torch.pow(error[self.next_particle_indices], 2).sum(dim=1).mean()

        if not train_flag:
            self.sum = targetpack.normalized_target[self.next_particle_indices].sum(dim=1).mean()
            loss_average = [self.loss.item(), torch.sqrt(self.loss).item()/self.sum.item()*100]
        else:
            self.sum = torch.pow(targetpack.normalized_target[self.next_particle_indices],2).sum(dim=1).mean()
            self.loss = torch.sqrt(self.loss)/self.sum
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
