
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from .agent import Agent
from ..utils.evaluation import entropy
from ..networks.critic import Critic


class DQN(Agent):

    def __init__(self,
                 obs_space,
                 action_space,
                 discount,
                 lr=1e-4,
                 seed=None):
        super(DQN, self).__init__(obs_space, action_space, discount, lr, seed)

        # epsilon decay
        self.initial_eps = 1.0
        self.end_eps = 1e-2
        self.eps_decay_period = 1000
        self.slope = (self.end_eps - self.initial_eps) / self.eps_decay_period
        self.eval_eps = 0.

        # loss function
        self.huber = nn.SmoothL1Loss()

        # Number of training iterations
        self.iterations = 0

        # After how many training steps 'snap' target to main network?
        self.target_update_freq = 100

        # Q-Networks
        self.Q = Critic(self.obs_space, self.action_space, seed).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)

        # Optimization
        self.optimizer = torch.optim.Adam(params=self.Q.parameters(), lr=self.lr)

    def policy(self, state, eval=False):

        # set networks to eval mode
        self.Q.eval()

        if eval:
            eps = self.eval_eps
        else:
            eps = max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        # epsilon greedy policy
        if self.rng.uniform(0, 1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_val = self.Q(state).cpu()
                return q_val.argmax().item(), q_val, np.nan
        else:
            return self.rng.integers(self.action_space), np.nan, np.nan

    def train(self, buffer, writer, minimum=None, maximum=None, use_probas=False):
        # Sample replay buffer
        state, action, next_state, reward, not_done = buffer.sample(minimum, maximum, use_probas)

        # set networks to train mode
        self.Q.train()
        self.Q_target.train()

        """
        # log state distribution
        if self.iterations % 1000 == 0:
            writer.add_histogram("train/states", state, self.iterations)
        """

        # Compute the target Q value
        with torch.no_grad():
            q_val = self.Q(next_state)
            next_action = q_val.argmax(dim=1, keepdim=True)
            target_Q = reward + not_done * self.discount * self.Q_target(next_state).gather(1, next_action)

        # Get current Q estimate
        current_Q = self.Q(state).gather(1, action)

        # Compute Q loss (Huber loss)
        Q_loss = self.huber(current_Q, target_Q)

        # log temporal difference error
        if self.iterations % 100 == 0:
            writer.add_scalar("train/TD-error", torch.mean(Q_loss).detach().cpu().item(), self.iterations)

        # Optimize the Q
        self.optimizer.zero_grad()
        Q_loss.backward()
        self.optimizer.step()

        self.iterations += 1
        # Update target network by full copy every X iterations.
        if self.iterations % self.target_update_freq == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def get_name(self) -> str:
        return "DeepQNetwork"

    def save_state(self) -> None:
        torch.save(self.Q.state_dict(), os.path.join("models", self.get_name() + "_Q.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", self.get_name() + "_optim.pt"))

    def load_state(self) -> None:
        self.Q.load_state_dict(torch.load(os.path.join("models", self.get_name() + "_Q.pt")))
        self.Q_target = copy.deepcopy(self.Q)
        self.optimizer.load_state_dict(torch.load(os.path.join("models", self.get_name() + "_optim.pt")))