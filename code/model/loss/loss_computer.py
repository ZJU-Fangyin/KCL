'''
Author: your name
Date: 2021-07-30 14:14:24
LastEditTime: 2021-07-30 19:20:48
LastEditors: your name
Description: In User Settings Edit
FilePath: /fangyin/contrastive-graph/model/loss/loss_computer.py
'''
import torch.nn as nn
import torch
import logging
logger = logging.getLogger()

class NCESoftmaxLoss(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
    
    def forward(self, similarity):
        batch_size = similarity.size(0) // 2
        label = torch.tensor([(batch_size + i) % (batch_size*2) for i in range(batch_size*2)]).to(self.device).long()
        loss = self.criterion(similarity, label)
        return loss


class FlatNCE(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
    
    def forward(self, similarity):
        pass

