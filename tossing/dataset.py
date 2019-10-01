import os
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TossingDataset(Dataset):
    def __init__(self, file_dir, in_traj_num=3, pre_traj_num=15, fps=30.0, sample_num=None):
        self.file_dir = file_dir
        self.in_traj_num = in_traj_num
        self.pre_traj_num = pre_traj_num
        self.fps = fps

        self.trial_list = []
        self.file_name_list = os.listdir(self.file_dir)
        
        # Sort the trial_list and get the first sample_num samples
        if sample_num:
            self.file_name_list = sorted(self.file_name_list)[:min(sample_num, len(self.file_name_list))]
        
        for file_name in self.file_name_list:
            # File path
            file_path = os.path.join(self.file_dir, file_name)

            # Load the file
            mat_file = sio.loadmat(file_path)

            # Read info
            x_locs = mat_file['position'][0]
            y_locs = mat_file['position'][1]
            
            # Generate physical solutions
            cur_x_gt = x_locs[:self.in_traj_num]
            cur_y_gt = y_locs[:self.in_traj_num]
            cur_all_gt = cur_x_gt.tolist() + cur_y_gt.tolist()

            future_x_gt = x_locs[self.in_traj_num : self.in_traj_num + self.pre_traj_num]
            future_y_gt = y_locs[self.in_traj_num : self.in_traj_num + self.pre_traj_num]
            future_all_gt = future_x_gt.tolist() + future_y_gt.tolist()

            future_locs_physics = get_physical_solution_future([cur_x_gt.tolist(), cur_y_gt.tolist()], 
                                                            pre_traj_num=self.pre_traj_num, 
                                                            fps=self.fps)
            self.trial_list.append({'file_path': file_path,
                                    'current_locs_gt': cur_all_gt,
                                    'future_locs_gt': future_all_gt,
                                    'future_locs_physics': future_locs_physics})
        

    def __len__(self):
        return len(self.trial_list)

    def __getitem__(self, idx):
        current_trail = self.trial_list[idx]
        
        sample = {
                  'current_locs_gt': torch.tensor(current_trail['current_locs_gt']).float(),
                  'future_locs_gt': torch.tensor(current_trail['future_locs_gt']).float(),
                  'future_locs_physics': torch.tensor(current_trail['future_locs_physics']).float() 
                 }
        
        return sample
    
    def get_file_list(self):
        return self.trial_list


def get_physical_solution_future(locs, pre_traj_num, fps=30.0):

    x_locs, y_locs = locs[0], locs[1]
    t = np.arange(0, 1/fps * (len(x_locs) + pre_traj_num), 1/fps)
    t_used = t[:len(x_locs)].reshape(-1, 1)
    t_future = t[len(x_locs):]
    
    # solve the least squares for x direction
    x_labels = np.array(x_locs).reshape(-1, 1)
    T = np.concatenate((np.ones(t_used.shape), t_used), axis=1)
    x_parameters = np.linalg.inv(T.T.dot(T)).dot(T.T).dot(x_labels)
    x_hat = t_future * x_parameters[1, 0] + x_parameters[0, 0]

    # solve the least squares for y direction
    g = -9.8
    y_labels = np.array(y_locs).reshape(-1, 1)
    T = np.concatenate((np.ones(t_used.shape), t_used), axis=1)
    y_parameters = np.linalg.inv(T.T.dot(T)).dot(T.T).dot(y_labels - 1/2 * g * t_used**2)
    y_hat = t_future**2 * 1/2 * g + t_future * y_parameters[1, 0] + y_parameters[0, 0]
    
    return x_hat.tolist() + y_hat.tolist()