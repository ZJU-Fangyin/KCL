
import torch.nn.functional as F
import torch.nn as nn
from dgl.nn.pytorch import GraphConv

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, gnn_norm='none', activation=None,
                 residual=True, batchnorm=True, dropout=0.):
        super(GCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv = GraphConv(in_feats=in_feats, out_feats=out_feats,
                                    norm=gnn_norm, activation=activation)
        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.graph_conv.reset_parameters()
        if self.residual:
            self.res_connection.reset_parameters()
        if self.bn:
            self.bn_layer.reset_parameters()

    def forward(self, g, feats):
        new_feats = self.graph_conv(g, feats)
        if self.residual:
            res_feats = self.activation(self.res_connection(feats))
            new_feats = new_feats + res_feats
        new_feats = self.dropout(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats=None, gnn_norm=None, activation=None,
                 residual=None, batchnorm=None, dropout=None):
        super(GCN, self).__init__()

        if hidden_feats is None:
            hidden_feats = [64, 64]

        n_layers = len(hidden_feats)
        if gnn_norm is None:
            gnn_norm = ['none' for _ in range(n_layers)]
        if activation is None:
            activation = [F.relu for _ in range(n_layers)]
        if residual is None:
            residual = [True for _ in range(n_layers)]
        if batchnorm is None:
            batchnorm = [True for _ in range(n_layers)]
        if dropout is None:
            dropout = [0. for _ in range(n_layers)]
        lengths = [len(hidden_feats), len(gnn_norm), len(activation),
                   len(residual), len(batchnorm), len(dropout)]
        assert len(set(lengths)) == 1, 'Expect the lengths of hidden_feats, gnn_norm, ' \
                                       'activation, residual, batchnorm and dropout to ' \
                                       'be the same, got {}'.format(lengths)

        self.hidden_feats = hidden_feats
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(GCNLayer(in_feats, hidden_feats[i], gnn_norm[i], activation[i],
                                            residual[i], batchnorm[i], dropout[i]))
            in_feats = hidden_feats[i]

    def reset_parameters(self):
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, feats):
        for gnn in self.gnn_layers:
            feats = gnn(g, feats)
        return feats
