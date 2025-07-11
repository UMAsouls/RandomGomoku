"""
自己対戦データセット
"""

import torch
from torch.utils.data import Dataset


class SelfPlayDataset(Dataset):
    """自己対戦データセット"""
    
    def __init__(self, states, policies, values):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.policies = torch.tensor(policies, dtype=torch.float32)
        self.values = torch.tensor(values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]
