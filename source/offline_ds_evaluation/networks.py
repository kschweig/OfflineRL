import torch
import torch.nn as nn
from abc import ABC


class BaseNet(nn.Module, ABC):

    def __init__(self, num_state, seed):
        super(BaseNet, self).__init__()

        # set seed
        torch.manual_seed(seed)

        self.num_hidden = 256

        self.base = nn.Sequential(
            nn.Linear(in_features=num_state, out_features=self.num_hidden),
            nn.SELU(),
            nn.Linear(in_features=self.num_hidden, out_features=self.num_hidden),
            nn.SELU(),
            nn.Linear(in_features=self.num_hidden, out_features=self.num_hidden),
            nn.SELU()
        )

        for param in self.parameters():
            if len(param.shape) == 1:
                torch.nn.init.constant_(param, 0)
            if len(param.shape) >= 2:
                torch.nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')

    def forward(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(dim=0)

        return self.base(state)


class BC(BaseNet):

    def __init__(self, num_state, num_actions, seed):
        super(BC, self).__init__(num_state, seed)

        self.out = nn.Linear(in_features=self.num_hidden, out_features=num_actions)

        for param in self.out.parameters():
            if len(param.shape) == 1:
                torch.nn.init.constant_(param, 0)
            if len(param.shape) >= 2:
                torch.nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')

    def forward(self, state):
        state = super(BC, self).forward(state)

        return self.out(state)

