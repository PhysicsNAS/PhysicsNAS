import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicalRegularization(nn.Module):
    def __init__(self, use_reg=True):
        super(PhysicalRegularization, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.use_reg = use_reg
        
    def physical_regularization(self, y_hat, x):
        
        # Initial energy
        initial_energy = torch.sum(torch.pow(x[...,:2], 2) * x[..., 4:6] * 0.5, dim=1)
        
        # Estimated energy
        est_energy = torch.sum(torch.pow(y_hat, 2) * x[..., 4:6] * 0.5, dim=1)
        
        # Energy difference 
        energy_diff = est_energy - initial_energy
        
        # Physical regularization
        physics_reg_loss = F.relu(energy_diff)
        
        return physics_reg_loss
    
    def forward(self, y_hat, y, x):
        if self.use_reg:
            return self.mse_loss(y_hat, y) + torch.mean(self.physical_regularization(y_hat, x))
        else:
            return self.mse_loss(y_hat, y)