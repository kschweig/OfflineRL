
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from .agent import Agent
from ..utils.evaluation import entropy
from ..networks.critic import Critic


class MCE(Agent):

    def __init__(self,
                 obs_space,
                 action_space,
                 discount,
                 lr,
                 seed=None):
        super(MCE, self).__init__(obs_space, action_space, discount, lr, seed)

        self.eval_eps = 0.

        # loss function
        self.huber = nn.SmoothL1Loss()

        # Number of training iterations
        self.iterations = 0

        # Q-Networks
        self.Q = Critic(self.obs_space, self.action_space, seed).to(self.device)

        # Optimization
        self.optimizer = torch.optim.Adam(params=self.Q.parameters(), lr=self.lr)

    def policy(self, state, eval=False):

        # set network to eval mode
        self.Q.eval()

        # epsilon greedy policy
        if self.rng.uniform(0, 1) > self.eval_eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_val = self.Q(state).cpu()
                return q_val.argmax().item(), q_val, np.nan
        else:
            return self.rng.integers(self.action_space), np.nan, np.nan

    def train(self, buffer, writer, minimum=None, maximum=None, use_probas=False):
        # Sample replay buffer
        state, action, next_state, reward, not_done = buffer.sample(minimum, maximum, use_probas, use_remaining_reward=True)

        # set network to train mode
        self.Q.train()

        # Get current Q estimate
        current_Q = self.Q(state).gather(1, action)

        # Compute Q loss (Huber loss)
        Q_loss = self.huber(current_Q, reward)

        # log temporal difference error
        if self.iterations % 100 == 0:
            writer.add_scalar("train/TD-error", torch.mean(Q_loss).detach().cpu().item(), self.iterations)

        # Optimize the Q
        self.optimizer.zero_grad()
        Q_loss.backward()
        self.optimizer.step()

        self.iterations += 1

    def get_name(self) -> str:
        return "Monte-Carlo Estimation"

    def save_state(self) -> None:
        torch.save(self.Q.state_dict(), os.path.join("models", self.get_name() + "_Q.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", self.get_name() + "_optim.pt"))

    def load_state(self) -> None:
        self.Q.load_state_dict(torch.load(os.path.join("models", self.get_name() + "_Q.pt")))
        self.Q_target = copy.deepcopy(self.Q)
        self.optimizer.load_state_dict(torch.load(os.path.join("models", self.get_name() + "_optim.pt")))