import torch
import torch.nn as nn
import torch.nn.functional as F

def get_mean_distance(locs_a, locs_b):
    vector_len = locs_a.shape[1]
    x_a = locs_a[:, :vector_len // 2]
    y_a = locs_a[:, vector_len // 2:]
    x_b = locs_b[:, :vector_len // 2]
    y_b = locs_b[:, vector_len // 2:]
    
    dif_x = (x_a - x_b) ** 2
    dif_y = (y_a - y_b) ** 2
    
    dif = dif_x + dif_y
    
    return torch.mean(torch.sqrt(dif))

class PhysicalRegularization(nn.Module):
    def __init__(self, use_reg=True):
        super(PhysicalRegularization, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.use_reg = use_reg
        
    def physical_regularization(self, y_hat, x):
        # y_hat: B K
        sample_num = y_hat.shape[-1] // 2

        # Get the initial speed
        initial_speed = x[..., 1] - x[..., 0]

        # Get x direction difference
        x_diff = y_hat[..., 1:sample_num] - y_hat[..., :sample_num - 1]

        # Physical regularization
        speed_sign = torch.sign(initial_speed).unsqueeze(1)

        physics_reg_loss = F.relu(x_diff * -speed_sign)
        return torch.mean(physics_reg_loss, dim=-1)
    
    def reg_indicator(self, y, x):
        gt_reg = self.physical_regularization(y, x)
        indicator = torch.zeros_like(gt_reg)
        indicator[gt_reg==0.0] = 1.0
        return indicator
    
    def forward(self, y_hat, y, x):
        if self.use_reg:
            return self.mse_loss(y_hat, y) + torch.mean(self.reg_indicator(y, x) * self.physical_regularization(y_hat, x))
        else:
            return self.mse_loss(y_hat, y)