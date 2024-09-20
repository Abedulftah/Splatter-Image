import torch
import torch.nn as nn

class DepthLossF(nn.Module):
    def __init__(self):
        super(DepthLossF, self).__init__()
    
    def forward(self, D_f):
        loss = torch.mean(torch.exp(-D_f))
        return loss


