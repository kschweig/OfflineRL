import numpy as np
import os
import copy
import torch
import torch.nn as nn
from .agent import Agent
from ..utils.evaluation import entropy
from ..networks.critic import QrCritic


class QRDQN(Agent):

    def __init__(self,
                 obs_space,
                 action_space,
                 discount,
                 lr=1e-4,
                 quantiles = 50,
                 seed=None):
        super(QRDQN, self).__init__(obs_space, action_space, discount, lr, seed)

        # epsilon decay
        self.initial_eps = 1.0
        self.end_eps = 1e-2
        self.eps_decay_period = 1000
        self.slope = (self.end_eps - self.initial_eps) / self.eps_decay_period
        self.eval_eps = 0.

        # Number of training iterations
        self.iterations = 0

        # After how many training steps 'snap' target to main network?
        self.target_update_freq = 100

        # Quantiles
        self.quantiles = quantiles
        self.quantile_tau = torch.FloatTensor([i / self.quantiles for i in range(1, self.quantiles + 1)]).to(self.device)

        # Q-Networks
        self.Q = QrCritic(self.obs_space, self.action_space, seed, quantiles).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)

        # huber loss
        self.huber = nn.SmoothL1Loss(reduction='none')

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

        # Compute the target Q value
        with torch.no_grad():
            target_Qs = self.Q(next_state)
            action_indices = torch.argmax(target_Qs.mean(dim=2), dim=1, keepdim=True)
            target_Qs = self.Q_target(next_state)
            target_Qs = target_Qs.gather(1, action_indices.unsqueeze(2).expand(-1, 1, self.quantiles))
            target_Qs = reward.unsqueeze(1) + not_done.unsqueeze(1) * self.discount * target_Qs

        # Get current Q estimate
        current_Qs = self.Q(state).gather(1, action.unsqueeze(2).expand(-1, 1,self.quantiles)).transpose(1, 2)

        # correct dimensions?
        assert target_Qs.shape == (buffer.batch_size, 1, self.quantiles), \
            f"Expected {(buffer.batch_size, 1, self.quantiles)}, was {target_Qs.shape} instead"
        assert current_Qs.shape == (buffer.batch_size, self.quantiles, 1), \
            f"Expected {(buffer.batch_size, self.quantiles, 1)}, was {current_Qs.shape} instead"

        # expand along singular dimensions
        target_Qs = target_Qs.expand(-1, self.quantiles, self.quantiles)
        current_Qs = current_Qs.expand(-1, self.quantiles, self.quantiles)

        # Compute TD error
        td_error = target_Qs - current_Qs
        assert td_error.shape == (buffer.batch_size, self.quantiles, self.quantiles), \
            f"Expected {(buffer.batch_size, self.quantiles, self.quantiles)}, was {td_error.shape} instead"

        # calculate loss through TD
        huber_l = self.huber(current_Qs, target_Qs)

        # calculate quantile loss
        quantile_loss = abs(self.quantile_tau - (td_error.detach() < 0).float()) * huber_l
        quantile_loss = quantile_loss.sum(dim=1).mean(dim=1).mean()

        # log temporal difference error and quantile loss
        if self.iterations % 100 == 0:
            writer.add_scalar("train/TD-error", torch.mean(huber_l).detach().cpu().item(), self.iterations)
            writer.add_scalar("train/quantile_loss", torch.mean(quantile_loss).detach().cpu().item(), self.iterations)

        # Optimize the Q
        self.optimizer.zero_grad()
        quantile_loss.backward()
        self.optimizer.step()

        self.iterations += 1
        # Update target network by full copy every X iterations.
        if self.iterations % self.target_update_freq == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def get_name(self) -> str:
        return "QuantileRegressionDeepQNetwork"

    def save_state(self) -> None:
        torch.save(self.Q.state_dict(), os.path.join("models", self.get_name() + "_Q.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", self.get_name() + "_optim.pt"))

    def load_state(self) -> None:
        self.Q.load_state_dict(torch.load(os.path.join("models", self.get_name() + "_Q.pt")))
        self.Q_target = copy.deepcopy(self.Q)
        self.optimizer.load_state_dict(torch.load(os.path.join("models", self.get_name() + "_optim.pt")))