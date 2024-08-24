import torch
import torch.nn as nn

class DepthLossF(nn.Module):
    def __init__(self, C):
        super(DepthLossF, self).__init__()
        self.C = C
    
    def forward(self, D1, D2):
        # Calculate the difference between the depth matrices
        diff = D1 - D2
        
        # Penalize when the difference is less than C
        loss_below_C = torch.relu(self.C - diff)
        
        # Penalize when the difference is greater than 2C
        loss_above_2C = torch.relu(diff - 2 * self.C)
        
        # Combine both penalties and average over all elements
        loss = torch.mean(loss_below_C + loss_above_2C)
        
        return loss