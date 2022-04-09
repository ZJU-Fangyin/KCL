import torch.nn as nn
from ..model_utils import get_activation_function
from ..layer.gcn import GCN
from ..layer.readout import WeightedSumAndMax
import pdb


class GCNNodeEncoder(nn.Module):
    def __init__(self, args):
        super(GCNNodeEncoder, self).__init__()
        activation = get_activation_function(args['activation'])

        self.gnn = GCN(in_feats=args['node_indim'],
                       hidden_feats=[args['hidden_feats']] * args['num_gnn_layers'],
                       gnn_norm=args['gnn_norm'],
                       activation=[activation] * args['num_gnn_layers'],
                       residual=[args['residual']] * args['num_gnn_layers'],
                       batchnorm=[args['batchnorm']] * args['num_gnn_layers'],
                       dropout=[args['dropout']] * args['num_gnn_layers'])
        gnn_out_feats = self.gnn.hidden_feats[-1]
        self.out_dim = gnn_out_feats

        self.node_emb = nn.Embedding(343, args['node_indim'])
        self.edge_emb = nn.Embedding(21, args['edge_indim'])

    def forward(self, bg):
        node_feats = self.gnn(bg, self.node_emb(bg.ndata['h']))
        return node_feats


class GCNEncoder(nn.Module):
    def __init__(self, args):
        super(GCNEncoder, self).__init__()
        activation = get_activation_function(args['activation'])

        self.gnn = GCN(in_feats=args['node_indim'],
                       hidden_feats=[args['hidden_feats']] * args['num_gnn_layers'],
                       gnn_norm=args['gnn_norm'],
                       activation=[activation] * args['num_gnn_layers'],
                       residual=[args['residual']] * args['num_gnn_layers'],
                       batchnorm=[args['batchnorm']] * args['num_gnn_layers'],
                       dropout=[args['dropout']] * args['num_gnn_layers'])
        gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.out_dim = 2 * gnn_out_feats


    def forward(self, bg):
        node_feats = self.gnn(bg, self.node_emb(bg.ndata['h']))
        graph_feats = self.readout(bg, node_feats)
        return graph_feats
