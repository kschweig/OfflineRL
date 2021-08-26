import torch
import torch.nn as nn


class Actor(nn.Module):

    def __init__(self, num_state, num_actions, seed):
        super(Actor, self).__init__()

        # set seed
        torch.manual_seed(seed)

        num_hidden = 256

        self.fnn = nn.Sequential(
            nn.Linear(in_features=num_state, out_features=num_hidden),
            nn.SELU(),
            nn.Linear(in_features=num_hidden, out_features=num_hidden),
            nn.SELU(),
            nn.Linear(in_features=num_hidden, out_features=num_hidden),
            nn.SELU(),
            nn.Linear(in_features=num_hidden, out_features=num_actions)
        )

        for param in self.parameters():
            if len(param.shape) == 1:
                torch.nn.init.constant_(param, 0)
            if len(param.shape) >= 2:
                torch.nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')

    def forward(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(dim=0)

        return self.fnn(state)
