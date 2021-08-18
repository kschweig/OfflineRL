import numpy as np
import torch


class ReplayBuffer():

    def __init__(self, obs_space, buffer_size, batch_size, seed=None):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.rng = np.random.default_rng(seed=seed)

        self.idx = 0
        self.current_size = 0

        self.state = np.zeros((self.buffer_size + 1, obs_space), dtype=np.float32)
        self.action = np.zeros((self.buffer_size + 1, 1), dtype=np.uint8)
        self.reward = np.zeros((self.buffer_size, 1))
        self.not_done = np.zeros((self.buffer_size, 1), dtype=np.bool_)

        # probas is uniform until updated by experiments
        self.probas = np.ones((self.buffer_size)) / self.buffer_size

    def add(self, state, action, reward, done):

        self.state[self.idx] = state
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.not_done[self.idx] = not done

        self.idx = (self.idx + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self, minimum=None, maximum=None, use_probas=False, use_remaining_reward=False, give_next_action=False):

        # we can set custom min/max to e.g. iterate over the dataset
        if minimum != None and maximum != None:
            ind = np.arange(minimum, maximum)
        else:
            ind = np.arange(0, self.current_size)

        # we can use custom sampling probabilities
        if use_probas:
            ind = self.rng.choice(ind, size=self.batch_size, replace=False, p=self.probas)
        else:
            ind = self.rng.choice(ind, size=self.batch_size, replace=False)

        if use_remaining_reward:
            reward = self.remaining_reward[ind]
        else:
            reward = self.reward[ind]

        if give_next_action:
            return (torch.FloatTensor(self.state[ind]).to(self.device),
                    torch.LongTensor(self.action[ind]).to(self.device),
                    torch.FloatTensor(self.state[ind+1]).to(self.device),
                    torch.LongTensor(self.action[ind+1]).to(self.device),
                    torch.FloatTensor(reward).to(self.device),
                    torch.FloatTensor(self.not_done[ind]).to(self.device)
                    )

        else:
            return (torch.FloatTensor(self.state[ind]).to(self.device),
                    torch.LongTensor(self.action[ind]).to(self.device),
                    torch.FloatTensor(self.state[ind+1]).to(self.device),
                    torch.FloatTensor(reward).to(self.device),
                    torch.FloatTensor(self.not_done[ind]).to(self.device)
                    )

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed=seed)

    def subset(self, minimum, maximum, retain_last=True):
        add = 1 if retain_last else 0
        self.state = self.state[minimum:maximum+add]
        self.action = self.action[minimum:maximum+add]
        self.reward = self.reward[minimum:maximum]
        self.not_done = self.not_done[minimum:maximum]

        self.idx = 0
        self.current_size = maximum - minimum

    def rand_subset(self, samples):
        ind = np.arange(0, self.buffer_size)
        ind = self.rng.choice(ind, size=samples, replace=False)

        self.reward = self.reward[ind]
        self.not_done = self.not_done[ind]

        np.asarray(ind.tolist().append(self.buffer_size + 1))

        self.state = self.state[ind]
        self.action = self.action[ind]

        self.idx = 0
        self.current_size = samples

    def mix(self, buffer, p_orig=0.5):
        # check if target buffer is big enough
        assert self.current_size >= buffer.current_size, \
            f"Target buffer too small, must be >= {self.current_size}, is {buffer.current_size}"

        buffer.subset(int(self.current_size * p_orig), self.current_size)
        self.subset(0, int(self.current_size * p_orig), retain_last=False)

        self.state = np.concatenate((self.state, buffer.state), axis=0)
        self.action = np.concatenate((self.action, buffer.action), axis=0)
        self.reward = np.concatenate((self.reward, buffer.reward), axis=0)
        self.not_done = np.concatenate((self.not_done, buffer.not_done), axis=0)

        self.current_size += buffer.current_size

    #####################################
    # Special functions for experiments #
    #####################################

    def calc_remaining_reward(self, discount=1):
        self.remaining_reward = np.zeros_like(self.reward)

        cum_reward = 0
        for i, d in reversed(list(enumerate(self.not_done))):
            if not d:
                cum_reward = 0
            cum_reward = cum_reward + self.reward[i]
            self.remaining_reward[i] = cum_reward
            cum_reward *= discount

