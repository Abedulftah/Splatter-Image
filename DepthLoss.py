import torch
import torch.nn as nn

class DepthLossF(nn.Module):
    def __init__(self):
        super(DepthLossF, self).__init__()
    
    def forward(self, D_f):
        D_f -= 1e-3
        loss = torch.mean(nn.functional.relu(-D_f))
        return loss


