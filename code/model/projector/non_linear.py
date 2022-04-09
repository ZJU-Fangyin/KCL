import torch.nn as nn

class NonLinearProjector(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_feats, in_feats//2),
            nn.ReLU(),
            nn.Linear(in_feats//2, in_feats)
        )

    def forward(self, features):
        return self.projector(features)