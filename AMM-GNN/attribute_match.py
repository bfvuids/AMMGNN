import torch
from torch import nn
from torch.nn import functional as F
import random

class Attr_match(nn.Module):
    def __init__(self, config):
        super(Attr_match, self).__init__()
        self.config = config[0]
        self.fea_size = config[1]
        self.vars = nn.ParameterList()

        if self.config == 4:
            w11 = nn.Parameter(torch.ones([1,16]))
            torch.nn.init.normal_(w11)
            self.vars.append(w11)
            w12 = nn.Parameter(torch.ones([16,500]))
            torch.nn.init.normal_(w12)
            self.vars.append(w12)
            w21 = nn.Parameter(torch.zeros([1,16]))
            torch.nn.init.kaiming_normal_(w21)
            w21 = nn.Parameter(w21/100)
            self.vars.append(w21)
            w22 = nn.Parameter(torch.zeros([16,500]))
            torch.nn.init.kaiming_normal_(w22)
            w22 = nn.Parameter(w22/100)
            self.vars.append(w22)
            w13 = nn.Parameter(torch.ones([16,1]))
            torch.nn.init.normal_(w13)
            self.vars.append(w13)
            w23 = nn.Parameter(torch.ones([16,1]))
            torch.nn.init.normal_(w23)
            self.vars.append(w23)


    def forward(self, all_features, x_spt, x_qry, task_num):

        sample_list = [[],[]]
        for listi in range(2):
            sample_list[listi] = random.sample(range(list(all_features.size())[0]),500)
        alpha_11 = torch.mm(self.vars[1],all_features[sample_list[0]]) + self.vars[4]
        alpha01 = torch.mm(self.vars[0],torch.tanh(alpha_11))
        alpha_12 = torch.mm(self.vars[1],all_features[sample_list[1]]) + self.vars[4]
        alpha02 = torch.mm(self.vars[0],torch.tanh(alpha_12))
        b_11 = torch.mm(self.vars[3],all_features[sample_list[0]]) + self.vars[5]
        b01 = torch.mm(self.vars[2],torch.tanh(b_11))
        b_12 = torch.mm(self.vars[3],all_features[sample_list[1]]) + self.vars[5]
        b02 = torch.mm(self.vars[2],torch.tanh(b_12))
        alpha = torch.abs((alpha01 + alpha02)/2)
        b = torch.abs((b01 + b02)/2)
        cos_sim = - torch.cosine_similarity(alpha01, alpha02, dim=1) - torch.cosine_similarity(b01, b02, dim=1) + 2
        for i in range(task_num):
            x_spt[i] = torch.mul(alpha,x_spt[i]) + b
            x_qry[i] = torch.mul(alpha,x_qry[i]) + b
        return x_spt, x_qry, cos_sim

    def zero_grad(self, vars=None): 
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()


    def parameters(self):
        return self.vars
