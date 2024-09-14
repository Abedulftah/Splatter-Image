import torch
import torch.nn as nn

class DepthLossF(nn.Module):
    def __init__(self):
        super(DepthLossF, self).__init__()
    
    def forward(self, D1, D2):
        diff = D1 - D2
        
        mask1 = (diff > 0).float()
        mask2 = (diff == 0).float() 
        loss = torch.mean(diff * mask1 + 1e-3 * mask2)
        
        return loss


