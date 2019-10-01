import torch
import torch.nn as nn

# Basic operations
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
class FCReLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCReLU, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim), 
            nn.BatchNorm1d(out_dim), # BN for fast training
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.operation(x)
    
class FCOut(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCOut, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        return self.operation(x)
    
class PhysicalForward(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PhysicalForward, self).__init__()
        self.operation = nn.Sequential(
            nn.Linear(in_dim, 4),
            nn.BatchNorm1d(4), # BN for fast training
        )
        
        self.out_dim = out_dim
        
    def physical_forward(self, parameters):
    
        v_a_1 = parameters[..., 0:1]
        v_b_1 = parameters[..., 1:2]

        m_a = parameters[..., 2:3]
        m_b = parameters[..., 3:4]

        v_a_f = (v_a_1 * (m_a - m_b) + 2 * v_b_1 * m_b) / (m_a + m_b)
        v_b_f = (v_b_1 * (m_b - m_a) + 2 * v_a_1 * m_a) / (m_a + m_b)

        return torch.cat((v_a_f, v_b_f), dim=1)

    def forward(self, x):
        parameters = self.operation(x)
        return self.physical_forward(parameters)

# A list of possible operations
operation_dict_diff_dim = {
    'fc_relu': lambda in_dim, out_dim: FCReLU(in_dim, out_dim),
}

operation_dict_same_dim = {
    'fc_relu': lambda in_dim, out_dim: FCReLU(in_dim, out_dim),
    'skip_connect': lambda in_dim, out_dim: Identity(),
}

operation_dict_diff_dim_out = {
    'fc_out': lambda in_dim, out_dim: FCOut(in_dim, out_dim),
    'phy_forward':  lambda in_dim, out_dim: PhysicalForward(in_dim, out_dim),
}

operation_dict_same_dim_out = {
    'fc_out': lambda in_dim, out_dim: FCOut(in_dim, out_dim),
    'skip_connect': lambda in_dim, out_dim: Identity(),
}