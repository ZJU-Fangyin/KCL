

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from dgl.nn.pytorch import NNConv
from ..layer.kmpnn import KMPNN

class MPNNGNN(nn.Module):
    def __init__(self, args):
        super(MPNNGNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(args['node_indim'], args['node_hidden_feats']),
            nn.ReLU()
        )
        self.num_step_message_passing = args['num_step_message_passing']
        edge_network = nn.Sequential(
            nn.Linear(args['edge_indim'], args['edge_hidden_feats']),
            nn.ReLU(),
            nn.Linear(args['edge_hidden_feats'], args['node_hidden_feats'] * args['node_hidden_feats'])
        )
        self.gnn_layer = NNConv(
            in_feats=args['node_hidden_feats'],
            out_feats=args['node_hidden_feats'],
            edge_func=edge_network,
            aggregator_type='sum'
        )
        self.gru = nn.GRU(args['node_hidden_feats'], args['node_hidden_feats'])
        self.out_dim = args['node_hidden_feats']

        self.node_emb = nn.Embedding(343, args['node_indim'])
        self.edge_emb = nn.Embedding(21, args['edge_indim'])

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node_feats[0].reset_parameters()
        self.gnn_layer.reset_parameters()
        for layer in self.gnn_layer.edge_func:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g):

        node_feats = self.node_emb(g.ndata['h'])
        edge_feats = self.edge_emb(g.edata['e'])

        node_feats = self.project_node_feats(node_feats) # (V, node_out_feats)
        hidden_feats = node_feats.unsqueeze(0)           # (1, V, node_out_feats)

        for _ in range(self.num_step_message_passing):
            node_feats = F.relu(self.gnn_layer(g, node_feats, edge_feats))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)
        return node_feats


class KMPNNGNN(nn.Module):
    def __init__(self, args, entity_emb, relation_emb):
        super(KMPNNGNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(args['node_indim'], args['node_hidden_feats']),
            nn.ReLU()
        )
        self.num_step_message_passing = args['num_step_message_passing']
        attn_fc = nn.Linear(2 * args['node_hidden_feats'], 1, bias=False)
        edge_network1 = nn.Sequential(
            nn.Linear(args['edge_indim'], args['edge_hidden_feats']),
            nn.ReLU(),
            nn.Linear(args['edge_hidden_feats'], args['node_hidden_feats'] * args['node_hidden_feats'])
        )
        edge_network2 = nn.Sequential(
            nn.Linear(args['edge_indim'], args['edge_hidden_feats']),
            nn.ReLU(),
            nn.Linear(args['edge_hidden_feats'], args['node_hidden_feats'] * args['node_hidden_feats'])
        )
        self.gnn_layer = KMPNN(
            in_feats=args['node_hidden_feats'],
            out_feats=args['node_hidden_feats'],
            attn_fc=attn_fc,
            edge_func1=edge_network1,
            edge_func2=edge_network2,
            aggregator_type='sum'
        )
        self.gru = nn.GRU(args['node_hidden_feats'], args['node_hidden_feats'])
        self.out_dim = args['node_hidden_feats']

        # self.node_emb = nn.Embedding(343, args['node_indim'])
        # self.edge_emb = nn.Embedding(21, args['edge_indim'])

        atom_emb = torch.randn((118, args['node_indim']))
        node_emb = torch.cat((atom_emb, entity_emb),0)
        bond_emb = torch.randn((4,args['edge_indim']))
        edge_emb = torch.cat((bond_emb, relation_emb),0)
        self.node_emb = nn.Embedding.from_pretrained(node_emb, freeze=False)
        self.edge_emb = nn.Embedding.from_pretrained(edge_emb, freeze=False)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node_feats[0].reset_parameters()
        self.gnn_layer.reset_parameters()
        for layer in self.gnn_layer.edge_func:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g):
        node_feats = self.node_emb(g.ndata['h'])
        edge_feats = self.edge_emb(g.edata['e'])

        node_feats = self.project_node_feats(node_feats) # (V, node_out_feats)
        hidden_feats = node_feats.unsqueeze(0)           # (1, V, node_out_feats)

        for _ in range(self.num_step_message_passing):
            node_feats = F.relu(self.gnn_layer(g, node_feats, edge_feats))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)
        return node_feats

