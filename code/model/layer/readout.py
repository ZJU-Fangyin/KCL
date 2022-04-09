
import dgl
import torch
import torch.nn as nn

from dgl.nn.pytorch import WeightAndSum

import torch as th
import torch.nn as nn
import numpy as np


from dgl.readout import sum_nodes, broadcast_nodes, softmax_nodes

class WeightedSumAndMax(nn.Module):
    def __init__(self, in_feats):
        super(WeightedSumAndMax, self).__init__()
        self.weight_and_sum = WeightAndSum(in_feats)
        self.out_dim = 2 * in_feats

    def forward(self, bg, feats):
        h_g_sum = self.weight_and_sum(bg, feats)
        with bg.local_scope():
            bg.ndata['h'] = feats
            h_g_max = dgl.max_nodes(bg, 'h')
        h_g = torch.cat([h_g_sum, h_g_max], dim=1)
        return h_g


class Set2Set(nn.Module):
    def __init__(self, input_dim, n_iters, n_layers):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = th.nn.LSTM(self.output_dim, self.input_dim, n_layers)
        self.reset_parameters()
        self.out_dim = self.output_dim

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        self.lstm.reset_parameters()

    def forward(self, graph, feat):
        with graph.local_scope():
            batch_size = graph.batch_size

            h = (feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
                 feat.new_zeros((self.n_layers, batch_size, self.input_dim)))

            q_star = feat.new_zeros(batch_size, self.output_dim)

            for _ in range(self.n_iters):
                q, h = self.lstm(q_star.unsqueeze(0), h)
                q = q.view(batch_size, self.input_dim)
                e = (feat * broadcast_nodes(graph, q)).sum(dim=-1, keepdim=True)
                graph.ndata['e'] = e
                alpha = softmax_nodes(graph, 'e')
                graph.ndata['r'] = feat * alpha
                readout = sum_nodes(graph, 'r')
                q_star = th.cat([q, readout], dim=-1)

            return q_star