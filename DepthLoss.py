import torch
import torch.nn as nn

class DepthLossF(nn.Module):
    def __init__(self):
        super(DepthLossF, self).__init__()
    
    def forward(self, D1, D2):
        loss = torch.mean(((D1 - D2) >= 0).float())
    
        return loss