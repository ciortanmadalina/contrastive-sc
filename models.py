"""
Original implementation of Contrastive-sc method
(https://github.com/ciortanmadalina/contrastive-sc)
By Madalina Ciortan (01/10/2020)
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
import torchvision

import models

class ContrastiveRepresentation(nn.Module):
    """
    Clustering network

    Args:
        nn ([type]): [description]
    """
    def __init__(self, dims, dropout = 0.8):
        super(ContrastiveRepresentation, self).__init__()
        self.dims = dims
        self.n_stacks = len(self.dims) #- 1
        enc = []
        for i in range(self.n_stacks - 1):
            if i == 0:
                enc.append(nn.Dropout(p =dropout))
            enc.append(nn.Linear(self.dims[i], self.dims[i+1]))
            enc.append(nn.BatchNorm1d(self.dims[i+1]))
            enc.append(nn.ReLU())

        enc = enc[:-2]
        self.encoder= nn.Sequential(*enc)
        self._reset_prams()

    def _reset_prams(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        return

    def forward(self, x):
        latent_out = self.encoder(x)
        latent_out = F.normalize(latent_out, dim = 1)
        return latent_out
    
class STClustering(nn.Module):
    """
    Clustering network

    Args:
        nn ([type]): [description]
    """
    def __init__(self, dims, t_alpha = 1):
        super(STClustering, self).__init__()
        self.t_alpha = t_alpha
        self.phase = "1"
        self.dims = dims
        self.n_stacks = len(self.dims) - 1
        enc = []
        for i in range(self.n_stacks - 1):
            enc.append(nn.Linear(self.dims[i], self.dims[i+1]))
            enc.append(nn.BatchNorm1d(self.dims[i+1]))
            enc.append(nn.ReLU())

        enc = enc[:-2]
        self.encoder= nn.Sequential(*enc)
        self.clustering = model.ClusterlingLayer(self.dims[-2], self.dims[-1])
        self._reset_prams()

    def _reset_prams(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        return
    def cal_latent(self, hidden, alpha):
        sum_y = torch.sum(torch.mul(hidden, hidden), dim=1)
        num = -2.0 * torch.matmul(hidden, torch.t(hidden)) + sum_y.view((-1, 1)) + sum_y
        num = num / alpha
        num = torch.pow(1.0 + num, -(alpha + 1.0) / 2.0)
        zerodiag_num = num - torch.diag(torch.diagonal(num))
        latent_p = torch.t(torch.t(zerodiag_num) / torch.sum(zerodiag_num, dim=1))
        return num, latent_p
    
    def target_dis(self, latent_p):
        latent_q = torch.t(torch.t(torch.pow(latent_p, 2))/torch.sum(latent_p, dim = 1))
        res = torch.t(torch.t(latent_q)/torch.sum(latent_q, dim =1))
        return res

    def forward(self, x):
        latent_out = self.encoder(x)
        if self.phase == "1":
            latent_out = F.normalize(latent_out, dim = 1)
        if self.phase == "2":
            normalized = F.normalize(latent_out, dim = 1)
            latent_dist1, latent_dist2 = self.clustering(latent_out)
        
            num, latent_p = self.cal_latent(latent_out, 1)
            latent_q = self.target_dis(latent_p)
            latent_p = latent_p + torch.diag(torch.diagonal(num))
            latent_q = latent_q + torch.diag(torch.diagonal(num))
            result = {
                "latent": latent_out,
                "latent_dist1": latent_dist1,
                "latent_dist2": latent_dist2,
                "latent_q": latent_q,
                "latent_p": latent_p,
                "num": num,
                "normalized": normalized

            }
            return result
        return latent_out
    


class MeanAct(nn.Module):
    def __init__(self, minval=1e-5, maxval=1e6):
        '''
        Init method.
        '''
        super().__init__() # init the base class
        self.minval = minval
        self.maxval = maxval

    def forward(self, inp):
        return torch.clamp(torch.exp(inp), self.minval, self.maxval)
    
class DispAct(nn.Module):
    def __init__(self, minval=1e-4, maxval=1e4):
        '''
        Init method.
        '''
        super().__init__() # init the base class
        self.minval = minval
        self.maxval = maxval

    def forward(self, inp):
        return torch.clamp(F.softplus(inp), self.minval, self.maxval)
    
    
class ClusterlingLayer(nn.Module):
    """
    Clustering layer to be applied on top of the representation layer.

    Args:
        nn ([type]): [description]
    """
    def __init__(self, in_features=10, out_features=10, alpha=1.0):
        super(ClusterlingLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.unsqueeze(1) - self.weight
        x = torch.mul(x, x)
        dist1 = torch.sum(x, dim=2)
        temp_dist1 = dist1 - torch.min(dist1, dim = 1)[0].view((-1, 1))
        q = torch.exp(-temp_dist1)
        q = torch.t(torch.t(q)/torch.sum(q, dim =1))
        q = torch.pow(q, 2)
        q = torch.t(torch.t(q)/torch.sum(q, dim =1))
        dist2 = dist1 * q
        return dist1, dist2


    def extra_repr(self):
        return 'in_features={}, out_features={}, alpha={}'.format(
            self.in_features, self.out_features, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)
    
def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)

def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) +np.inf, x)


def _nelem(x):
    nelem = torch.sum((~torch.isnan(x)).type(torch.FloatTensor))
    res = torch.where(torch.equal(nelem, 0.), 1., nelem)
    return res

def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return torch.divide(torch.sum(x), nelem)


def clustering_loss(result, gamma = 0.001, alpha = 0.001):
    """
    Clustering loss used in our method on top of the representation layer.

    Args:
        result ([type]): [description]
        gamma (float, optional): [description]. Defaults to 0.001.
        alpha (float, optional): [description]. Defaults to 0.001.

    Returns:
        [type]: [description]
    """
    cross_entropy_loss = - torch.sum(result['latent_q'] * torch.log(result['latent_p']))
    kmeans_loss = torch.mean(torch.sum(result['latent_dist2'], dim=1))
    entropy_loss = -torch.sum(result['latent_q'] * torch.log(result['latent_q']))
    kl_loss = cross_entropy_loss- entropy_loss
    total_loss =  alpha * kmeans_loss + gamma * kl_loss
    return total_loss
