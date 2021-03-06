from source.agents.agent import Agent
from source.utils.utils import entropy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np


class REM(Agent):

    def __init__(self, params):

        super(REM, self).__init__(params.action_space, params.frame_stack)

        self.device = params.device
        self.params = params

        # Determine network type
        self.Q = Network(self.frames, self.action_space, num_heads=params.heads).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)

        # Determine optimizer
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(self.Q.parameters(), lr=params.lr, eps=params.eps)
        elif params.optimizer == "RMSProp":
            self.optimizer = optim.RMSProp(self.Q.parameters(), lr=params.lr, eps=params.eps)
        else:
            raise ValueError("Optimizer not implemented")

        # loss function
        self.huber = nn.SmoothL1Loss()

        # Number of training iterations
        self.iterations = 0

        # how to update target network
        self.maybe_update_target = self.soft_target_update if params.soft_target_update else self.copy_target_update

        # slope
        self.slope = (self.params.end_eps - self.params.initial_eps) / self.params.eps_decay_period

    def get_name(self):
        return "REM"

    def policy(self, state, eval=False, eps=None):
        if eps == None:
            if eval:
                eps = self.params.eval_eps
            else:
                eps = max(self.slope * self.iterations + self.params.initial_eps, self.params.end_eps)

        # epsilon greedy policy
        if np.random.uniform(0, 1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                if eval:
                    q_values = self.Q.evaluate(state)
                    return int(q_values.argmax(dim=1)), entropy(q_values), float(q_values.max(dim=1)[0])
                else:
                    q_values = self.Q.forward(state)
                    return int(q_values.argmax(dim=1)), np.nan, np.nan
        else:
            return np.random.randint(self.action_space), np.nan, np.nan

    def train(self, replay_buffer):
        # Sample replay buffer
        state, action, next_state, reward, done = replay_buffer.sample()

        # Compute the target Q value
        with torch.no_grad():
            target_Q = reward + done * self.params.discount * self.Q_target(next_state).max(1, keepdim=True)[0]

        # Get current Q estimate
        current_Q = self.Q(state).gather(1, action)

        # Compute Q loss (Huber loss)
        Q_loss = self.huber(current_Q, target_Q)

        # Optimize the Q
        self.optimizer.zero_grad()
        Q_loss.backward()
        self.optimizer.step()

        # Update target network either continuously by soft update or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()

    def soft_target_update(self):
        """Soft update model parameters.
        ??_target = ??*??_local + (1 - ??)*??_target
        """
        for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(self.params.tau * param.data + (1.0 - self.params.tau) * target_param.data)

    def copy_target_update(self):
        """
        Hard update model parameters, "snap" target policy to local policy
        :return:
        """
        if self.iterations % self.params.target_update_freq == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def save_state(self, online, run):
        mode = "online" if online else "offline"
        torch.save(self.Q.state_dict(), os.path.join("data", self.params.experiment, "models",
                                                     self.get_name() + "_" + mode + f"_{run}_Q.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("data", self.params.experiment, "models",
                                                             self.get_name() + "_" + mode + f"_{run}_optimizer.pt"))

    def load_state(self, online, run):
        mode = "online" if online else "offline"
        self.Q.load_state_dict(torch.load(
            os.path.join("data", self.params.experiment, "models", self.get_name() + "_" + mode + f"_{run}_Q.pt")))
        self.Q_target = copy.deepcopy(self.Q)
        self.optimizer.load_state_dict(torch.load(os.path.join("data", self.params.experiment, "models",
                                                               self.get_name() + "_" + mode + f"_{run}_optimizer.pt")))


class Network(nn.Module):

    def __init__(self, frames, num_actions, num_heads):
        super(Network, self).__init__()

        self.num_actions = num_actions
        self.num_heads = num_heads

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=frames, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=7 * 7 * 64, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions*num_heads)
        )

    def forward(self, state):

        if len(state.shape) == 3:
            state = state.unsqueeze(dim=0)

        features = self.net(state).reshape(state.shape[0], self.num_heads, self.num_actions)

        alphas = torch.rand(self.num_heads).to(device=state.device)
        alphas /= torch.sum(alphas)
        # insert dimensions to multiply in correct dimension
        alphas = alphas.unsqueeze(0).unsqueeze(2)

        return torch.sum(features * alphas, dim=1)

    def evaluate(self, state):
        """
        Need evaluate, as we want to
        :param state:
        :return:
        """
        if len(state.shape) == 3:
            state = state.unsqueeze(dim=0)

        features = self.net(state).reshape(state.shape[0], self.num_heads, self.num_actions)

        return torch.mean(features, dim=1)


