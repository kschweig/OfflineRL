import numpy as np
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from .networks import BC
from .training import update, evaluate
from .datasets import BCSet
from .utils import entropy, BColors
from hyperloglog import HyperLogLog
import matplotlib.pyplot as plt
from math import sqrt
import copy


class Evaluator:

    def __init__(self,
                 environment: str,
                 buffer_type: str,
                 states:np.ndarray,
                 actions:np.ndarray,
                 rewards:np.ndarray,
                 dones:np.ndarray,
                 workers=0,
                 seed=42,
                 num_actions=None):

        assert len(states.shape) == 2, f"States must be of dimension (ds_size, feature_size), were ({states.shape})"
        if len(actions.shape) == 1:
            actions = actions.reshape(-1, 1)
        assert len(actions.shape) == 2, f"Actions must be of dimension (ds_size, 1), were ({actions.shape})"
        if len(rewards.shape) == 1:
            rewards = rewards.reshape(-1, 1)
        assert len(rewards.shape) == 2, f"Rewards must be of dimension (ds_size, 1), were ({actions.shape})"
        if len(dones.shape) == 1:
            dones = dones.reshape(-1, 1)
        assert len(dones.shape) == 2, f"Dones must be of dimension (ds_size, 1), were ({actions.shape})"

        # task information
        self.environment = environment
        self.buffer_type = buffer_type

        # Dataset, last state and actions are meaningless
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones

        # auxiliary parameters
        self.workers = workers
        self.seed = seed

        # could be that dataset contains not every action, then one can pass the correct number of actions
        self.num_actions = num_actions if num_actions is not None else np.max(self.actions) + 1

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # behavioral cloning network
        self.behavioral_trained = False
        self.behavioral = BC(num_state=self.states.shape[1], num_actions=self.num_actions, seed=self.seed).to(device)

    def evaluate(self, state_limits=None, action_limits=None,
                 epochs=10, batch_size=64, lr=1e-3,
                 subsample=1., verbose=False):

        assert 0 <= subsample <= 1, f"subsample must be in [0;1] but is {subsample}."

        self.train_behavior_policy(epochs, batch_size, lr, verbose)

        returns = self.get_returns()
        sparsities = self.get_sparsities()
        ep_lengths = self.get_episode_lengths()
        entropies = self.get_bc_entropy()

        unique_states = self.get_unique_states(limits=state_limits)
        unique_state_actions = self.get_unique_state_actions(limits=action_limits)

        if verbose:
            print("-"*50)
            print("Min / Mean / Max Return: \t\t", f"{round(np.min(returns), 2)} / {round(np.mean(returns), 2)} "
                                                 f"/ {round(np.max(returns), 2)}")
            print("Unique States: \t", f"{unique_states}")
            print("Unique State-Actions: \t", f"{unique_state_actions}")
            print("Min / Mean / Max Entropy: \t", f"{round(np.min(entropies), 2)} / {round(np.mean(entropies), 2)} "
                                                  f"/ {round(np.max(entropies), 2)}")
            print("Min / Mean / Max Sparsity: \t", f"{round(np.min(sparsities), 2)} / "
                                                   f"{round(np.mean(sparsities), 2)} "
                                                   f"/ {round(np.max(sparsities), 2)}")
            print("Min / Mean / Max Episode Length: \t", f"{round(np.min(ep_lengths), 2)} / "
                                                         f"{round(np.mean(ep_lengths), 2)} "
                                                         f"/ {round(np.max(ep_lengths), 2)}")
            print("-" * 50)

        return returns, unique_states, unique_state_actions, entropies, sparsities, ep_lengths

    def get_returns(self):

        rewards, ep_reward = list(), 0

        for i, done in enumerate(self.dones):
            ep_reward += self.rewards[i].item()
            if done:
                rewards.append(ep_reward)
                ep_reward = 0

        return rewards

    @staticmethod
    def get_normalized_rewards(rewards, random_reward, optimal_reward):
        normalized_reward = []
        for reward in rewards:
            normalized_reward.append((reward - random_reward) / (optimal_reward - random_reward))
        return normalized_reward

    def get_sparsities(self):

        sparsity, num_not_obtained = list(), list()

        for i, done in enumerate(self.dones):
            num_not_obtained.append(self.rewards[i].item() == 0)
            if done:
                sparsity.append(np.mean(num_not_obtained))
                num_not_obtained = list()

        return sparsity

    def get_episode_lengths(self):

        lengths, ep_length = list(), 0

        for i, done in enumerate(self.dones):
            ep_length += 1
            if done:
                lengths.append(ep_length)
                ep_length = 0

        return lengths

    def get_bc_entropy(self):
        if not self.behavioral_trained:
            print(BColors.WARNING + "Attention, behavioral policy was not trained before calling get_bc_entropy!" + BColors.ENDC)

        entropies = []
        dl = DataLoader(BCSet(states=self.states, actions=self.actions), batch_size=512, drop_last=False,
                        shuffle=False, num_workers=self.workers)

        for x, _ in dl:
            x = x.to(next(self.behavioral.parameters()).device)
            entropies.extend(entropy(self.behavioral(x)))

        # calculate entropy
        entropies = np.asarray(entropies)

        return entropies

    def get_unique_states(self, states=None, limits=None):
        if states is None:
            states = copy.deepcopy(self.states)

        for axis in range(len(states[0])):
            if limits is None:
                axmin, axmax = np.min(states[:, axis]), np.max(states[:, axis])
            else:
                axmin, axmax = limits[axis*2:axis*2+2]

            states[:, axis] = np.digitize(states[:, axis],
                                          np.linspace(axmin, axmax, num=100))
        states.astype(int)

        hll = HyperLogLog(0.01)
        for state in tqdm(states,
                          desc=f"Search for Unique States in whole dataset ({self.environment} @ {self.buffer_type})",
                          total=len(states)):
            hll.add(",".join([str(s) for s in state]))

        return len(hll)

    def get_unique_state_actions(self, states=None, actions=None, limits=None):
        if states is None:
            states = copy.deepcopy(self.states)
        if actions is None:
            actions = copy.deepcopy(self.actions)

        states = np.concatenate((states, actions), axis=1)

        return self.get_unique_states(states, limits)

    def get_unique_states_exact(self):
        unique = set()
        for i in tqdm(range(len(self.dones)),
                            desc=f"Search exact for Unique States in whole dataset ({self.environment} @ {self.buffer_type})",
                            total=len(self.dones)):
            unique.add(",".join([str(s) for s in self.states[i]]))
        return len(unique)

    def get_unique_state_actions_exact(self):
        unique = set()
        states = copy.deepcopy(self.states + self.actions)
        for i in tqdm(range(len(self.dones)),
                            desc=f"Search exact for Unique State-Action pairs in whole dataset ({self.environment} @ {self.buffer_type})",
                            total=len(self.dones)):
            unique.add(",".join([str(s) for s in states[i]]))
        return len(unique)

    def train_behavior_policy(self, epochs=10, batch_size=64, lr=1e-3, verbose=False):

        dl = DataLoader(BCSet(states=self.states, actions=self.actions), batch_size=batch_size, drop_last=True,
                        shuffle=True, num_workers=self.workers)
        optimizer = Adam(self.behavioral.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss()

        if verbose:
            print(f"Inital loss:", evaluate(self.behavioral, dl, loss))

        for ep in tqdm(range(epochs), desc=f"Training Behavioral Policy ({self.environment} @ {self.buffer_type})"):
            errs = update(self.behavioral, dl, loss, optimizer)

            if verbose:
                print(f"Epoch: {ep+1}, loss: {np.mean(errs)}")

        self.behavioral_trained = True

