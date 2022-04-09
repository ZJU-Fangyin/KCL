'''
Author: your name
Date: 2021-07-30 14:14:24
LastEditTime: 2021-07-30 14:43:26
LastEditors: your name
Description: In User Settings Edit
FilePath: /fangyin/contrastive-graph/model/loss/pn_generator.py
'''
import torch.nn as nn
import torch
from torch.autograd import Variable
import dgl
import random
import math
import pdb

from torch.nn.functional import dropout

class NodeDropoutNoisePNGenerator():
    def __init__(self, dropout: float=0.1) -> None:
        self.dropout = dropout
    
    def apply(self, dgl_graph):
        nodes = dgl_graph.nodes().tolist()
        dgl_subgraph = dgl.node_subgraph(dgl_graph, random.sample(nodes, math.ceil(len(nodes)*(1-self.dropout))))
        return dgl_subgraph

class NodeMaskNoisePNGenerator():
    def __init__(self, dropout: float=0.1) -> None:
        self.dropout = dropout
    
    def apply(self, dgl_graph):
        nodes = dgl_graph.nodes().tolist()
        mask_nodes = random.sample(nodes, math.floor(len(nodes)*self.dropout))
        dgl_subgraph = dgl.node_subgraph(dgl_graph, dgl_graph.nodes())
        dgl_subgraph.ndata['h'][mask_nodes] = torch.ones(dgl_subgraph.ndata['h'][mask_nodes].shape, dtype=torch.float)
        return dgl_subgraph
        
class BernoulliDropoutNoisePNGenerator(nn.Module):
    def __init__(self, dropout: float=0.1):
        super(BernoulliDropoutNoisePNGenerator, self).__init__()
        self.dropout_anchor = nn.Dropout(dropout)
        self.dropout_positive = nn.Dropout(dropout)
    
    def forward(self, emb):
        z_i = self.dropout_anchor(emb)
        z_j = self.dropout_positive(emb)

        return z_i, z_j

class BernoulliDropoutDimensionPNGenerator(nn.Module):
    def __init__(self, dim_hidden, dropout: float=0.1):
        super(BernoulliDropoutDimensionPNGenerator, self).__init__()
        self.dim_hidden = dim_hidden
        self.dropout = dropout
        dropout_dim = random.sample(list(range(self.dim_hidden)), math.floor(self.dropout*self.dim_hidden))
        self.dropout_mask = torch.ones((1, self.dim_hidden))
        self.dropout_mask[0, dropout_dim] = torch.zeros(self.dropout_mask[0, dropout_dim].shape)

    def update(self):
        dropout_dim = random.sample(list(range(self.dim_hidden)), math.floor(self.dropout*self.dim_hidden))
        self.dropout_mask = torch.ones((1, self.dim_hidden))
        self.dropout_mask[0, dropout_dim] = torch.zeros(self.dropout_mask[0, dropout_dim].shape)
    
    def forward(self, emb):
        return emb, emb * self.dropout_mask


class GaussTimeNoisePNGenerator(nn.Module):
    def __init__(self, device, alpha=1.0):
        super(GaussTimeNoisePNGenerator, self).__init__()
        self.device = device
        self.alpha = torch.Tensor([alpha]).to(self.device)
    
    def forward(self, emb):
        z_i = torch.randn(emb.size(), device=emb.device) * (self.alpha)
        z_i.requires_grad = False
        z_j = torch.randn(emb.size(), device=emb.device) * (self.alpha)
        z_i.requires_grad = False
        return emb * z_i, emb * z_j 

        
class GaussPlusNoisePNGenerator(nn.Module):
    def __init__(self, device, alpha=1.0):
        super(GaussPlusNoisePNGenerator, self).__init__()
        self.device = device
        self.alpha = torch.Tensor([alpha]).to(self.device)
    
    def forward(self, emb):
        z_i = torch.randn(emb.size(), device=emb.device) * (self.alpha)
        z_i.requires_grad = False
        z_j = torch.randn(emb.size(), device=emb.device) * (self.alpha)
        z_i.requires_grad = False

        return emb + z_i, emb + z_j



