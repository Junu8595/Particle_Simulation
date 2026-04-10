import torch
import torch.nn as nn

from collections import namedtuple
    
class MLP(nn.Module):
    def __init__(self,
                 name,
                 num_hidden_layer,
                 input_size,
                 hidden_size,
                 output_size,
                 norm = False,
                 bias = True,
                 residual = True):
        
        super(MLP, self).__init__()
        
        self.name = name
        self.residual = residual
        input_layer = [nn.Linear(input_size, hidden_size,bias=bias),
                       nn.ReLU()]
        hidden_layer = []
        for _ in range(num_hidden_layer):
            hidden_layer.append(nn.Linear(hidden_size,hidden_size,bias=bias))
            hidden_layer.append(nn.ReLU())

        output_layer = [nn.Linear(hidden_size, output_size, bias=bias)]
        if norm:
            output_layer.append(nn.LayerNorm(output_size))
        
        self.net = nn.Sequential(*input_layer, *hidden_layer, *output_layer)

    def forward(self, x):
        return self.net(x)
    

class graph_net(nn.Module):
    def __init__(self,
                 network_attribute_pack):
        super(graph_net, self).__init__()
        self.name = 'graph_network'
        self.sub_nets = nn.ModuleList()
        self.sub_nets_name = []

        for attribute in network_attribute_pack:
            if attribute:
                network, param = self.build_net(attribute)
                self.sub_nets_name.append(param[0])
                setattr(self.sub_nets,param[0],network)


    def build_net(self, attribute):
        if attribute.multi_mlp_cnt < 0:
            print("INVALID MULTI MLP CNT")
            input()
        
        network = nn.ModuleList()
        for i in range(attribute.multi_mlp_cnt):
            network.append(MLP(attribute.name,
                               attribute.n_layers - 2,
                               attribute.input_size,
                               attribute.hidden_size,
                               attribute.output_size,
                               attribute.norm,
                               attribute.bias,
                               attribute.residual))
            
        return network, [*attribute]
    
    def forward(self, network, x, residual_x = None, step = None):
        if len(network) == 1:
            idx = 0
        else:
            idx = int(step)
        if step == 9:
            a = 1
        if network[idx].residual:
            res = residual_x.clone()
            return network[idx](x) + res
        else:
            return network[idx](x)
        

    
