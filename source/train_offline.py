import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils.evaluation import evaluate
from .utils.utils import get_agent, make_env


def train_offline(experiment, envid, agent_type="DQN", buffer_type="er", discount=0.95, transitions=100000,
                  batch_size=128, lr=1e-4,
                  use_run=1, run=1, seed=42,
                  use_subset=False, lower=None, upper=None,
                  use_progression=False, buffer_size=None,
                  use_remaining_reward=False):
    # over how many episodes do we take average and how much gradient updates to next
    evaluate_every = transitions // 1000 // 5

    env = make_env(envid)
    obs_space = len(env.observation_space.high)
    agent = get_agent(agent_type, obs_space, env.action_space.n, discount, lr, seed)

    # load saved buffer
    with open(os.path.join("data", f"ex{experiment}", f"{envid}_run{use_run}_{buffer_type}.pkl"), "rb") as f:
        buffer = pickle.load(f)

    # configure buffer
    buffer.batch_size = batch_size

    #######################
    # experiment specific #
    #######################

    if use_remaining_reward: buffer.calc_remaining_reward(discount=discount)
    if use_subset: buffer.subset(lower, upper)

    #######################

    # seeding
    env.seed(seed)
    np.random.seed(seed)
    buffer.set_seed(seed)
    torch.manual_seed(seed)

    writer = SummaryWriter(
        log_dir=os.path.join("runs", f"ex{experiment}", f"{envid}", f"{buffer_type}", f"{agent_type}", f"run{run}"))

    all_rewards = []

    for iter in tqdm(range(transitions), desc=f"{agent_type} ({envid}) {buffer_type}, run {run}"):
        if use_progression:
            minimum = max(0, iter - buffer_size)
            maximum = max(batch_size, iter)
        else:
            minimum = None
            maximum = None

        agent.train(buffer, writer, maximum, minimum)

        if (iter + 1) % evaluate_every == 0:
            all_rewards = evaluate(env, agent, writer, all_rewards)


    # save returns of offline training
    os.makedirs(os.path.join("results", "raw", f"ex{experiment}"), exist_ok=True)
    with open(os.path.join("results", "raw", f"ex{experiment}", f"{envid}_userun{use_run}_run{run}_{buffer_type}.csv"),
              "w") as f:
        for r in all_rewards:
            f.write(f"{r}\n")

    return agent
