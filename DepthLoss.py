import torch
import torch.nn as nn

class DepthLossF(nn.Module):
    def __init__(self, C):
        super(DepthLossF, self).__init__()
        self.C = C
    
    def forward(self, D1, D2):

        diff = D1 - D2

        loss_below_C = torch.relu(self.C - diff)
        loss = torch.mean(loss_below_C)
        
        return loss