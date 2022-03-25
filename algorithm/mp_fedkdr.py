from pathlib import Path
from .mp_fedbase import MPBasicServer, MPBasicClient
import torch.nn as nn
import numpy as np

import torch
import os
import copy


def kernel_matrix(input):
    """
    input is of shape [batch_size, d]
    with d is an arbitary number, the dimention of features
    """
    # np.savetxt("input.txt", delimiter=',', X=input.cpu().numpy())
    std = torch.std(input, dim=0)
    input = input.T.unsqueeze(-1)
    _ , batch_size, _ = input.shape
    base_vol = input * torch.ones((batch_size, batch_size))
    diff_wise = base_vol - base_vol.transpose(1,2)
    diff_wise_norm = torch.norm(diff_wise, dim=0)
    # diff_wise_norm /= (2 * torch.mean(std)**2)
    output = torch.exp(-diff_wise_norm) * (1 - torch.eye(batch_size))
    return output


def proba_matrix(kernel_matrix):
    """
    kernel_matrix is the symmetric matrix, output from kernel_matrix function
    """
    return kernel_matrix/torch.sum(kernel_matrix, dim=1, keepdim=True)


def KL_divergence(proba_matrix_teacher, proba_matrix_student):
    """
    Compute the KL divergence of 2 proba matrices
    """
    temp = torch.nan_to_num(proba_matrix_teacher/proba_matrix_student, 1)
    log_temp = torch.log(temp)
    return torch.sum(proba_matrix_teacher * log_temp)


class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        
    def finish(self, model_path):
        if not Path(model_path).exists():
            os.system(f"mkdir -p {model_path}")
        task = self.option['task']
        torch.save(self.model.state_dict(), f"{model_path}/{self.name}_{self.num_rounds}_{task}.pth")
        pass
    
    def run(self):
        super().run()
        # self.finish(f"algorithm/fedrl_utils/baseline/{self.name}")
        for i in range(len(self.clients)):
            self.clients[i].dump_kd_loss(i, f"algorithm/kd_utils/{self.name}")
        return
    
    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        models, train_losses = self.communicate(self.selected_clients,pool)
        if not self.selected_clients: 
            return
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        self.model = self.aggregate(models, p = [1.0 for cid in self.selected_clients])
        return


class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.lossfunc = nn.CrossEntropyLoss()
        self.kd_factor = 0.05
        self.kd_loss_record = []
        
        
    def train(self, model, device):
        model = model.to(device)
        model.train()
        
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
                
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        kd_losses = []
        for iter in range(self.epochs):
            batch_kl_loss = []
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss, kl_loss = self.get_loss(model, src_model, batch_data, device)
                loss = loss + kl_loss
                loss.backward()
                optimizer.step()
                batch_kl_loss.append(kl_loss.cpu().item())
            # print(f"epochs {iter}: kl_loss {np.mean(batch_kl_loss)}")
            kd_losses.append(np.mean(batch_kl_loss))
        
        self.kd_loss_record.append(np.mean(kd_losses))
        self.kd_factor = min(self.kd_factor * 1.05, 0.25)
        return
    
    
    def data_to_device(self, data,device):
        return data[0].to(device), data[1].to(device)


    def get_loss(self, model, src_model, data, device):
        tdata = self.data_to_device(data, device)
        output_s, representation_s = model.pred_and_rep(tdata[0])   # Student
        _ , representation_t = src_model.pred_and_rep(tdata[0])     # Teacher
        
        Kx = kernel_matrix(representation_t.detach().cpu())    # Teacher
        Ky = kernel_matrix(representation_s.detach().cpu())    # Student

        P_t = proba_matrix(Kx)  # Teacher prob
        Q_s = proba_matrix(Ky)  # Student prob

        kl_loss = KL_divergence(P_t, Q_s)   # KL divergence
        loss = self.lossfunc(output_s, tdata[1])
        return loss, self.kd_factor * kl_loss
    
    
    def dump_kd_loss(self, id, folder):
        if not Path(folder).exists():
            os.system(f"mkdir -p {folder}")
            
        np.savetxt(f"{folder}/kd_loss_{id}.loss", X=np.array(self.kd_loss_record), delimiter=',')
        return