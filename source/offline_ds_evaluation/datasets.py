import numpy as np
import torch
from torch.utils.data import Dataset


class BCSet(Dataset):

    def __init__(self, states, actions):
        super(BCSet, self).__init__()

        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, item):
        return torch.FloatTensor(self.states[item]), torch.LongTensor(self.actions[item])