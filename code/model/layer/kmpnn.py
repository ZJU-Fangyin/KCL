"""Torch Module for NNConv layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
from os import fchdir
import torch as th
from torch import nn
from torch.nn import init

from dgl import function as fn
import torch.nn.functional as F
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
import pdb
import numpy as np
import time


class KMPNN(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 attn_fc,
                 edge_func1,
                 edge_func2,
                 aggregator_type='mean',
                 residual=False,
                 bias=True):
        super(KMPNN, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.attn_fc = attn_fc
        self.edge_func1 = edge_func1
        self.edge_func2 = edge_func2
        if aggregator_type == 'sum':
            self.reducer = fn.sum
        elif aggregator_type == 'mean':
            self.reducer = fn.mean
        elif aggregator_type == 'max':
            self.reducer = fn.max
        else:
            raise KeyError('Aggregator type {} not recognized: '.format(aggregator_type))
        self._aggre_type = aggregator_type
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized using Glorot uniform initialization
        and the bias is initialized to be zero.
        """
        gain = init.calculate_gain('relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = th.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'attn_e': F.leaky_relu(a)}

    def message_func1(self, edges):
        return {'m1' : edges.src['h'] * edges.data['w1'], 'attn_e1':edges.data['attn_e'], 'z1': edges.src['z']}

    def message_func2(self, edges):
        return {'m2' : edges.src['h'] * edges.data['w2'], 'attn_e2':edges.data['attn_e'], 'z2': edges.src['z']}

    def reduce_func1(self, nodes):
        alpha = F.softmax(nodes.mailbox['attn_e1'], dim=1).unsqueeze(-1)
        h = th.sum(alpha * nodes.mailbox['m1'], dim=1)
        return {'neigh1': h} 

    def reduce_func2(self, nodes):
        alpha = F.softmax(nodes.mailbox['attn_e2'], dim=1).unsqueeze(-1)
        h = th.sum(alpha * nodes.mailbox['m2'], dim=1)
        return {'neigh2': h} 

    def forward(self, graph, feat, efeat):

        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)

            # (n, d_in, 1)
            graph.srcdata['h'] = feat_src.unsqueeze(-1)
    
            # (n, d_in, d_out)
            graph.edata['w1'] = self.edge_func1(efeat).view(-1, self._in_src_feats, self._out_feats)
            graph.edata['w2'] = self.edge_func2(efeat).view(-1, self._in_src_feats, self._out_feats)
            
            graph.ndata['z'] = feat_src
            graph.apply_edges(self.edge_attention)
            # pdb.set_trace()
            # (n, d_in, d_out)
            edges1 = th.nonzero(graph.edata['etype']==0).squeeze(1).int() # bonds
            edges2 = th.nonzero(graph.edata['etype']==1).squeeze(1).int() # rels
            
            # graph.send_and_recv(edges1, fn.u_mul_e('h', 'w1', 'm'), self.reducer('m', 'neigh1')) 
            graph.send_and_recv(edges1, self.message_func1, self.reduce_func1)
            graph.send_and_recv(edges2, self.message_func2, self.reduce_func2) 
            rst1 = graph.dstdata['neigh1'].sum(dim=1)
            rst2 = graph.dstdata['neigh2'].sum(dim=1) # (n, d_out)
            rst = rst1 + rst2 # (n, d_out)

            # residual connection
            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)
            # bias
            if self.bias is not None:
                rst = rst + self.bias
            return rst
