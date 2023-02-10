import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Union

class TableData(Dataset):
    
    def __init__(self, data: pd.DataFrame, label_idx: Union[None, int],
                normalize: bool = True):
        #Read csv file
        self.data = data.to_numpy()
        
        #Collect labels (if any)
        if label_idx:
            self.labels = self.data[:, label_idx]
            self.labels = torch.from_numpy(self.labels)
            self.data = np.concatenate((self.data[:, 0:label_idx],
                                       self.data[:, label_idx + 1:]), axis = 1)
        else:
            self.labels = None
        
        #Normalize data to mean zero standard deviation 1
        if normalize:
            self.data =(self.data - self.data.mean(axis = 0)) / self.data.std(axis=0)
        
        #Convert to torch tensor
        self.data = torch.from_numpy(self.data)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if isinstance(self.labels, torch.Tensor):
            return self.data[idx, :], self.labels[idx]
        return self.data[idx, :]