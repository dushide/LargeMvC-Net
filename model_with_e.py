import torch
import torch.nn.functional as F
import torch.nn as nn
import scipy.sparse as sp
from util.utils import normalization, standardization
from scipy.linalg import svd


class DBONet_with_E(nn.Module):
    def __init__(self, m, n_view, blocks, theta1, theta2, Z_init, D_init, E_init, device):
        super(DBONet_with_E, self).__init__()
        self.blocks = blocks
        self.device = device
        self.n_view = n_view
        self.m = m
        self.Z_init = torch.from_numpy(Z_init).float().to(device)
        self.D_init = [torch.tensor(d, dtype=torch.float32).to(device) for d in D_init]
        self.E_init = [torch.tensor(e, dtype=torch.float32).to(device) for e in E_init]
        self.theta1 = nn.Parameter(torch.FloatTensor([theta1]), requires_grad=True).to(device)
        self.theta2 = nn.Parameter(torch.FloatTensor([theta2]), requires_grad=True).to(device)
        self.U = nn.Linear(m, m).to(device)
        self.S = nn.Linear(m, m).to(device)

    # L1 regularization
    def soft_threshold(self, u):
        return F.selu(u - self.theta1) - F.selu(-1.0 * u - self.theta1)

    # L21 regularization
    def soft_threshold2(self, x):
            nw = torch.norm(x)
            if nw > self.theta2:
                x = (nw - 1 / self.theta2) * x / nw
            else:
                x = torch.zeros_like(x)
            return (x)


    def forward(self, features):
        Z = list()
        D_list = list()
        E_list = list()
        Z.append(self.Z_init)
        for i in range(self.n_view):
            exec('D{} = list()'.format(i))
            exec('D{}.append(self.D_init[{}])'.format(i, i))
            exec('E{} = list()'.format(i))
            exec('E{}.append(self.E_init[{}])'.format(i, i))
        for i in range(0, self.blocks):
            z = torch.zeros_like(self.Z_init)
            for j in range(self.n_view):
            # update z
                input1 = self.U(Z[-1].t()).t()
                exec('input2 = torch.mm(D{}[-1].t(), features[{}] - E{}[-1])'.format(j, j, j))
                exec('input2 = self.S(input2.t()).t()')
                exec('z += input1 + input2')
            Z.append(self.soft_threshold(z / self.n_view))
            # update D and E
            for k in range(self.n_view):
                exec('C = torch.mm(features[{}], Z[-1].t()) - torch.mm(E{}[-1], Z[-1].t())'.format(k, k))
                exec('C_numpy = C.detach().cpu().numpy()')
                exec('U, _, Vt = svd(C_numpy, full_matrices=False)')
                exec('U = torch.tensor(U, dtype=torch.float32, device=features[{}].device)'.format(k))
                exec('Vt = torch.tensor(Vt, dtype=torch.float32, device=features[{}].device)'.format(k))
                exec('D_i = torch.mm(U, Vt)')
                exec('D{}.append(D_i)'.format(k))
            for l in range(self.n_view):
                exec('E_i = features[{}] - torch.mm(D{}[-1], Z[-1])'.format(l, l))
                exec('E{}.append(self.soft_threshold2(E_i.t()).t())'.format(l))
        for i in range(self.n_view):
            exec('D_list.append(D{}[-1])'.format(i))
            exec('E_list.append(E{}[-1])'.format(i))
        return Z[-1], D_list, E_list

