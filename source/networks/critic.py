import torch
import torch.nn as nn


class Critic(nn.Module):

    def __init__(self, num_state, num_actions, seed, n_estimates=1):
        super(Critic, self).__init__()

        # set seed
        torch.manual_seed(seed)

        self.num_actions = num_actions
        num_hidden = 256

        self.backbone = nn.Sequential(
            nn.Linear(in_features=num_state, out_features=num_hidden),
            nn.SELU(),
            nn.Linear(in_features=num_hidden, out_features=num_hidden),
            nn.SELU(),
            nn.Linear(in_features=num_hidden, out_features=num_hidden),
            nn.SELU()
        )

        self.out = nn.Linear(in_features=num_hidden, out_features=num_actions * n_estimates)

        for param in self.parameters():
            if len(param.shape) == 1:
                torch.nn.init.constant_(param, 0)
            if len(param.shape) >= 2:
                torch.nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')

    def forward(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(dim=0)

        state = self.backbone(state)

        return self.out(state)


class RemCritic(Critic):

    def __init__(self, num_state, num_actions, seed, heads):
        super(RemCritic, self).__init__(num_state, num_actions, seed, heads)

        self.heads = heads

    def forward(self, state):
        state = super(RemCritic, self).forward(state)

        if self.training:
            alphas = torch.rand(self.heads).to(device=state.device)
            alphas /= torch.sum(alphas)

            return torch.sum(state.view(len(state), self.heads, self.num_actions) * alphas.view(1, -1, 1), dim=1)
        else:
            return torch.mean(state.view(len(state), self.heads, self.num_actions), dim=1)


class UncertaintyCritic(Critic):

    def __init__(self, num_state, num_actions, seed, heads):
        super(UncertaintyCritic, self).__init__(num_state, num_actions, seed, heads)

        self.heads = heads

    def forward(self, state):
        state = super(UncertaintyCritic, self).forward(state)

        if self.training:
            return state.view(len(state), self.heads, self.num_actions)
        else:
            # pessimistic estimate of the qvalue for selection
            q_std = torch.std(state.view(len(state), self.heads, self.num_actions), dim=1)
            qval = torch.mean(state.view(len(state), self.heads, self.num_actions), dim=1)
            return qval, q_std


class QrCritic(Critic):

    def __init__(self, num_state, num_actions, seed, quantiles):
        super(QrCritic, self).__init__(num_state, num_actions, seed, quantiles)

        self.quantiles = quantiles

    def forward(self, state):
        state = super(QrCritic, self).forward(state)

        state = state.reshape(len(state), self.num_actions, self.quantiles)

        if self.training:
            return state
        return torch.mean(state, dim=2)



