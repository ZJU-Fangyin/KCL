def load_classifier(args, exp_configure, config):

    if exp_configure['model'] == 'GCNv2':
        if args['cls'] == 'linear':
            classifier = nn.Linear(2 * exp_configure['gnn_hidden_feats'], exp_configure['n_tasks'])
        else:
            classifier = nn.Sequential(
                nn.Dropout(config['predictor_dropout']),
                nn.Linear(2 * exp_configure['gnn_hidden_feats'], config['predictor_hidden_feats']),
                nn.LeakyReLU(),
                nn.BatchNorm1d(config['predictor_hidden_feats']),
                nn.Linear(config['predictor_hidden_feats'], exp_configure['n_tasks']),
            )

import torch.nn as nn
class NonLinearPredictor(nn.Module):
    def __init__(self, in_feats, out_feats, config):
        super().__init__()
        self.dropout = nn.Dropout(config['predictor_dropout'])
        self.linear1 = nn.Linear(in_feats, config['predictor_hidden_feats'])
        self.activation = nn.GELU()
        self.batch_normal = nn.BatchNorm1d(config['predictor_hidden_feats'])
        self.linear2 = nn.Linear(config['predictor_hidden_feats'], out_feats)
    
    def forward(self, features):
        emb = self.dropout(features)
        emb = self.batch_normal(self.activation(self.linear1(emb)))
        emb = self.linear2(emb)
        return emb