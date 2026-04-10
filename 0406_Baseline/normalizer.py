import torch
import torch.nn as nn
import numpy as np

class online_normalizer(nn.Module):
    
    def __init__(self, name, size, max_accumulations = 10**6, std_epsilon = 1e-6, saving_path = './', device = 'cuda:0'):

        super(online_normalizer, self).__init__()

        self.name = name
        self.size = size
        self.device = device

        self.max_accumulations = max_accumulations
        self.std_epsilon = torch.tensor(std_epsilon, dtype=torch.float32, requires_grad=False, device=self.device)
        self.acc_cnt = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=self.device)[None,None]
        self.num_accumulations = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=self.device)[None,None]
        self.acc_sum = torch.zeros((1,size), dtype=torch.float32, requires_grad=False, device=self.device)
        self.acc_square = torch.zeros((1,size), dtype=torch.float32, requires_grad=False, device=self.device)

        self.saving_path = saving_path

        self.freeze = False

    def forward(self, batch, accumulate):
        if accumulate == True and self.freeze == False:
            if self.num_accumulations < self.max_accumulations:
                if batch.shape[0] > 0:
                    self.accumulate_stats(batch.detach())

        batch = (batch - self.cal_mean().to(self.device)) / self.cal_std().to(self.device)
        return batch


    def accumulate_stats(self, batch):
        cnt = batch.shape[0]
        batch_sum = torch.sum(batch, axis = 0, keepdim=True)
        batch_square = torch.sum(batch**2, axis = 0, keepdim=True)

        self.acc_sum += batch_sum
        self.acc_square += batch_square
        self.acc_cnt += cnt
        self.num_accumulations += 1

    def cal_mean(self):
        safe_cnt = torch.maximum(self.acc_cnt, torch.tensor(1.0, dtype=torch.float32, requires_grad=False, device=self.acc_cnt.device))
        return self.acc_sum / safe_cnt
    
    def cal_std(self):
        safe_cnt = torch.maximum(self.acc_cnt, torch.tensor(1.0, dtype=torch.float32, requires_grad=False, device=self.acc_cnt.device))
        std = torch.sqrt(self.acc_square / safe_cnt - self.cal_mean()**2)
        std = torch.nan_to_num(torch.maximum(std, self.std_epsilon), self.std_epsilon)
        return std
    
    """ saving the normalizing parameters into numpy array files """
    def save_variables(self):
        if self.num_accumulations <= self.max_accumulations and self.freeze == False:    
            save_variables = torch.cat((self.acc_cnt, self.acc_sum, self.acc_square, self.num_accumulations), dim=1).detach().cpu().numpy()
            np.save((self.saving_path + self.name + "_variables_test.npy"), save_variables)
            
    """ loads normalizer parameters from numpy array files """
    def load_normalizer(self, load_path):
        values = torch.from_numpy(np.load(load_path + self.name + "_variables_test.npy")).detach() 
        self.acc_cnt = values[:, 0, None].long().detach().to(self.device)
        self.acc_sum = values[:, 1:1+self.size].float().detach().to(self.device)
        self.acc_square = values[:, 1+self.size:1+2*self.size].float().detach().to(self.device)
        self.num_accumulations = values[:, -1, None].float().detach().to(self.device)

        self.freeze = True

    def inverse(self, batch):
        return batch * self.cal_std().to(batch.device) + self.cal_mean().to(batch.device)

