import numpy as np
import gym
import gym_minigrid
import gym_minatar
from gym_minigrid.wrappers import FullyObsWrapper
from .wrappers import FlatImgObsWrapper, RestrictMiniGridActionWrapper, MinAtarObsWrapper, RestrictMinAtarActionWraper
from ..agents.dqn import DQN
from ..agents.rem import REM
from ..agents.qrdqn import QRDQN
from ..agents.bcq import BCQ
from ..agents.mce import MCE
from ..agents.crr import CRR
from ..agents.cql import CQL
from ..agents.bc import BehavioralCloning
from ..agents.bve import BVE
from ..agents.random import Random


def get_agent(agent_type, obs_space, num_actions, discount, lr, seed):
    if agent_type == "DQN":
        return DQN(obs_space, num_actions, discount, lr, seed=seed)
    elif agent_type == "REM":
        return REM(obs_space, num_actions, discount, lr, heads=200, seed=seed)
    elif agent_type == "QRDQN":
        return QRDQN(obs_space, num_actions, discount, lr, quantiles=50, seed=seed)
    elif agent_type == "BCQ":
        return BCQ(obs_space, num_actions, discount, lr, seed=seed)
    elif agent_type == "MCE":
        return MCE(obs_space, num_actions, discount, lr, seed=seed)
    elif agent_type == "CRR":
        return CRR(obs_space, num_actions, discount, lr, seed=seed)
    elif agent_type == "CQL":
        return CQL(obs_space, num_actions, discount, lr, seed=seed)
    elif agent_type == "BC":
        return BehavioralCloning(obs_space, num_actions, discount, lr, seed=seed)
    elif agent_type == "BVE":
        return BVE(obs_space, num_actions, discount, lr, seed=seed)
    elif agent_type == "Random":
        return Random(obs_space, num_actions, discount, lr, seed=seed)
    else:
        print(BColors.WARNING + "Attention, using random agent!" + BColors.ENDC)
        return Random(obs_space, num_actions, discount, lr, seed=seed)


def make_env(envid):
    env = gym.make(envid)
    if "MiniGrid" in envid:
        env = FlatImgObsWrapper(RestrictMiniGridActionWrapper(env))
    if "MinAtar" in envid:
        env = MinAtarObsWrapper(env)
    return env


def cosine_similarity(s1, s2):
    assert len(s1.shape) == 1 and len(s2.shape) == 1, \
        f"s1 and s2 must be vectors, found shapes {s1.shape} and {s2.shape}"
    return cosine_similarity_n12(s1, s2, np.linalg.norm(s1), np.linalg.norm(s2))


def cosine_similarity_n2(s1, s2, n2):
    assert len(s1.shape) == 1 and len(s2.shape) == 1, \
        f"s1 and s2 must be vectors, found shapes {s1.shape} and {s2.shape}"
    return cosine_similarity_n12(s1, s2, np.linalg.norm(s1), n2)


def cosine_similarity_n12(s1, s2, n1, n2):
    assert len(s1.shape) == 1 and len(s2.shape) == 1, \
        f"s1 and s2 must be vectors, found shapes {s1.shape} and {s2.shape}"
    return np.sum(s1 * s2) / n1 / n2


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
