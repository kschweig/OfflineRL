import os
import h5py
import numpy as np
import urllib.request
from tqdm import tqdm


HOPPER_RANDOM_SCORE = -20.272305
HALFCHEETAH_RANDOM_SCORE = -280.178953
WALKER_RANDOM_SCORE = 1.629008

HOPPER_EXPERT_SCORE = 3234.3
HALFCHEETAH_EXPERT_SCORE = 12135.0
WALKER_EXPERT_SCORE = 4592.3

MAX_EPISODE_STEPS = 1000

"""
UNDISCOUNTED_POLICY_RETURNS = {
    'halfcheetah-medium': 3985.8150261686337,
    'halfcheetah-random': -199.26067391425954,
    'halfcheetah-expert': 12330.945945279545,
    'hopper-medium': 2260.1983114487352,
    'hopper-random': 1257.9757846810203,
    'hopper-expert': 3624.4696022560997,
    'walker2d-medium': 2760.3310101980005,
    'walker2d-random': 896.4751989935487,
    'walker2d-expert': 4005.89370727539,
}
"""

SOURCES = {"halfcheetah-medium-replay-v0": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_mixed.hdf5",
           "walker2d-medium-replay-v0": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker_mixed.hdf5",
           "hopper-medium-replay-v0": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_mixed.hdf5",

           "halfcheetah-random-v0": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_random.hdf5",
           "walker2d-random-v0": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_random.hdf5",
           "hopper-random-v0": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_random.hdf5",

           "halfcheetah-medium-v0": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_medium.hdf5",
           "walker2d-medium-v0": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_medium.hdf5",
           "hopper-medium-v0": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_medium.hdf5",

           "halfcheetah-expert-v0": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_expert.hdf5",
           "walker2d-expert-v0": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_expert.hdf5",
           "hopper-expert-v0": "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_expert.hdf5",
           }

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def load(name, link):
    os.makedirs("data", exist_ok=True)
    h5path = os.path.join('.', 'data', name.replace('-v0', '.hdf5'))
    if not os.path.exists(h5path):
        urllib.request.urlretrieve(link, h5path)

    dataset = dict()
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                dataset[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                dataset[k] = dataset_file[k][()]
    return dataset["observations"], dataset["actions"], dataset["rewards"], \
           np.invert(np.logical_or(dataset["terminals"], dataset["timeouts"]))


if __name__ == "__main__":

    for name, link in tqdm(SOURCES.items()):
        states, actions, rewards, not_dones = load(name, link)

        print(name)
        print(states.shape, actions.shape, rewards.shape, not_dones.shape)

