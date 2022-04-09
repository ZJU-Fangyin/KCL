import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeaveLayer(nn.Module):
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_node_hidden_feats=50,
                 edge_node_hidden_feats=50,
                 node_out_feats=50,
                 node_edge_hidden_feats=50,
                 edge_edge_hidden_feats=50,
                 edge_out_feats=50,
                 activation=F.relu):
        super(WeaveLayer, self).__init__()

        self.activation = activation

        # Layers for updating node representations
        self.node_to_node = nn.Linear(node_in_feats, node_node_hidden_feats)
        self.edge_to_node = nn.Linear(edge_in_feats, edge_node_hidden_feats)
        self.update_node = nn.Linear(
            node_node_hidden_feats + edge_node_hidden_feats, node_out_feats)

        # Layers for updating edge representations
        self.left_node_to_edge = nn.Linear(node_in_feats, node_edge_hidden_feats)
        self.right_node_to_edge = nn.Linear(node_in_feats, node_edge_hidden_feats)
        self.edge_to_edge = nn.Linear(edge_in_feats, edge_edge_hidden_feats)
        self.update_edge = nn.Linear(
            2 * node_edge_hidden_feats + edge_edge_hidden_feats, edge_out_feats)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.node_to_node.reset_parameters()
        self.edge_to_node.reset_parameters()
        self.update_node.reset_parameters()
        self.left_node_to_edge.reset_parameters()
        self.right_node_to_edge.reset_parameters()
        self.edge_to_edge.reset_parameters()
        self.update_edge.reset_parameters()

    def forward(self, g, node_feats, edge_feats, node_only=False):

        g = g.local_var()

        # Update node features
        node_node_feats = self.activation(self.node_to_node(node_feats))
        g.edata['e2n'] = self.activation(self.edge_to_node(edge_feats))
        g.update_all(fn.copy_edge('e2n', 'm'), fn.sum('m', 'e2n'))
        edge_node_feats = g.ndata.pop('e2n')
        new_node_feats = self.activation(self.update_node(
            torch.cat([node_node_feats, edge_node_feats], dim=1)))

        if node_only:
            return new_node_feats

        # Update edge features
        g.ndata['left_hv'] = self.left_node_to_edge(node_feats)
        g.ndata['right_hv'] = self.right_node_to_edge(node_feats)
        g.apply_edges(fn.u_add_v('left_hv', 'right_hv', 'first'))
        g.apply_edges(fn.u_add_v('right_hv', 'left_hv', 'second'))
        first_edge_feats = self.activation(g.edata.pop('first'))
        second_edge_feats = self.activation(g.edata.pop('second'))
        third_edge_feats = self.activation(self.edge_to_edge(edge_feats))
        new_edge_feats = self.activation(self.update_edge(
            torch.cat([first_edge_feats, second_edge_feats, third_edge_feats], dim=1)))

        return new_node_feats, new_edge_feats

class WeaveGNN(nn.Module):
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 num_layers=2,
                 hidden_feats=50,
                 activation=F.relu):
        super(WeaveGNN, self).__init__()

        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gnn_layers.append(WeaveLayer(node_in_feats=node_in_feats,
                                                  edge_in_feats=edge_in_feats,
                                                  node_node_hidden_feats=hidden_feats,
                                                  edge_node_hidden_feats=hidden_feats,
                                                  node_out_feats=hidden_feats,
                                                  node_edge_hidden_feats=hidden_feats,
                                                  edge_edge_hidden_feats=hidden_feats,
                                                  edge_out_feats=hidden_feats,
                                                  activation=activation))
            else:
                self.gnn_layers.append(WeaveLayer(node_in_feats=hidden_feats,
                                                  edge_in_feats=hidden_feats,
                                                  node_node_hidden_feats=hidden_feats,
                                                  edge_node_hidden_feats=hidden_feats,
                                                  node_out_feats=hidden_feats,
                                                  node_edge_hidden_feats=hidden_feats,
                                                  edge_edge_hidden_feats=hidden_feats,
                                                  edge_out_feats=hidden_feats,
                                                  activation=activation))
                                                
    def reset_parameters(self):
        """Reinitialize model parameters."""
        for layer in self.gnn_layers:
            layer.reset_parameters()

    def forward(self, g, node_feats, edge_feats, node_only=True):
        for i in range(len(self.gnn_layers) - 1):
            node_feats, edge_feats = self.gnn_layers[i](g, node_feats, edge_feats)
        return self.gnn_layers[-1](g, node_feats, edge_feats, node_only)