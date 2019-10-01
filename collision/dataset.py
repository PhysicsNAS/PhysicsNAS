import os
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CollisionDataset(Dataset):
    def __init__(self, file_dir, sample_num=None):
        self.file_dir = file_dir
        self.file_name_list = os.listdir(self.file_dir)
        
        # Sort the trial_list and get the first sample_num samples
        if sample_num:
            self.file_name_list = sorted(self.file_name_list)[:min(sample_num, len(self.file_name_list))]
        

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, idx):
        cur_file_name = self.file_name_list[idx]
        
        # File path
        file_path = os.path.join(self.file_dir, cur_file_name)
        
        # Load the file
        mat_file = sio.loadmat(file_path)
        
        # Read info
        v_1 = mat_file['v_i'][0]
        v_2 = mat_file['v_i'][1]
        mass = mat_file['mass'][0]
        v_t = mat_file['v_t'][0]
        v_phy = mat_file['v_phy'][0]
        distance = mat_file['distance'][0]
        
        sample = {
                  'x': torch.tensor(np.concatenate((v_1, v_2, mass, distance))).float(),
                  'y': torch.tensor(v_t).float(),
                  'y_phy': torch.tensor(v_phy).float() 
                 }
        
        return sample
    
    def get_file_list(self):
        return self.file_name_list