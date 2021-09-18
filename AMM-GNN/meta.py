import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from learner import Learner
from utils import f1
from itertools import combinations
from attribute_match import Attr_match
import os
import sys

class Meta(nn.Module):
    def __init__(self, args, config, all_label):
        super(Meta, self).__init__()
        self.update_lr = args.update_lr  
        self.meta_lr = args.meta_lr       
        self.n_way = args.n_way         
        self.k_spt = args.k_spt           
        self.k_qry = args.k_qry          
        self.task_num = args.task_num     
        self.update_step = args.update_step 
        self.update_step_test = args.update_step_test 
        self.net = Learner(config)
        self.match = Attr_match([4,config[0][1][1]])
        self.meta_optim =  optim.Adam([
                {'params': self.net.parameters(), 'weight_decay': 1e-4},
                {'params': self.match.parameters(), 'lr': 0.001}
                ], lr=self.meta_lr)
        self.label = all_label

    def clip_grad_by_norm_(self, grad, max_norm):  
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)
        return total_norm/counter

    def forward_less(self, x_spt, y_spt, x_qry, y_qry, y_idx, all_features): 
        task_num = self.task_num
        querysz = self.n_way * self.k_qry
        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):
            if i == 0:
                x_spt, x_qry, cos_sim = self.match(all_features[0],x_spt,x_qry,task_num)
            losses_q[-1] = losses_q[-1] + 0.1 * cos_sim
            logits = self.net(x_spt[i], vars=None, bn_training=True) 
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters()))) 
            with torch.no_grad():     
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True) 
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q                   
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct     

            with torch.no_grad():       
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct
            
            for k in range(1, self.update_step):   
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q
                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item() 
                    corrects[k + 1] = corrects[k + 1] + correct

        loss_q = losses_q[-1] / task_num
        self.meta_optim.zero_grad() 
        loss_q.backward()
        self.meta_optim.step()
        accs = np.array(corrects) / (querysz * task_num)
        return accs