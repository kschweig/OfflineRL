from source.train_online import train_online
from source.train_offline import train_offline
from source.offline_ds_evaluation.evaluator import Evaluator
from source.offline_ds_evaluation.metrics_manager import MetricsManager
from multiprocessing import Pool
import os
import copy
import pickle
import numpy as np
import argparse


# project parameters
env = "MiniGrid-Dynamic-Obstacles-8x8-v0"
discount = 0.95
buffer_types = ["random", "mixed", "er", "noisy", "fully"]
agent_types = ["BC", "BVE", "MCE", "DQN", "QRDQN", "REM", "BCQ", "CQL", "CRR"]
multiple_useruns = 5
# experiment parameters
experiment = 106
seed = 42
# hyperparameters for online training
behavioral = "DQN"
buffer_size = 50000
transitions_online = 100000
# hyperparameters for offline training
transitions_offline = 5 * transitions_online
batch_size = 128
lr = [1e-4] * len(agent_types)


def create_ds(args):
    train_online(experiment=experiment, agent_type=behavioral, discount=discount, envid=env,
                 transitions=transitions_online, buffer_size=buffer_size,
                 run=args, seed=seed)

def train(args):
    use_run, run, dataset = args

    for a, agent in enumerate(agent_types):
        for bt in range(len(buffer_types)):
            if 0 < dataset != bt:
                continue

            train_offline(experiment=experiment, envid=env, agent_type=agent, buffer_type=buffer_types[bt],
                          discount=discount, transitions=transitions_offline, batch_size=batch_size, lr=lr[a],
                          use_run=use_run, run=run, seed=seed+run, use_remaining_reward=(agent == "MCE"))

def assess_env(args):
    use_run = args

    with open(os.path.join("data", f"ex{experiment}", f"{env}_run{use_run}_er.pkl"), "rb") as f:
        buffer = pickle.load(f)
    state_limits = []
    for axis in range(len(buffer.state[0])):
        state_limits.append(np.min(buffer.state[:, axis]))
        state_limits.append(np.max(buffer.state[:, axis]))
    action_limits = copy.deepcopy(state_limits)
    action_limits.append(np.min(buffer.action))
    action_limits.append(np.max(buffer.action))

    results, returns, actions, entropies, sparsities, episode_lengts = [], [], [], [], [], []
    mm = MetricsManager(experiment)

    for buffer_type in buffer_types:
        with open(os.path.join("data", f"ex{experiment}", f"{env}_run{use_run}_{buffer_type}.pkl"), "rb") as f:
            buffer = pickle.load(f)

        evaluator = Evaluator(env, buffer_type, buffer.state, buffer.action, buffer.reward,
                              np.invert(buffer.not_done))

        if env == 'CartPole-v1' or env == 'MountainCar-v0':
            rets, us, usa, ents, sps, epls = evaluator.evaluate(state_limits=state_limits, action_limits=action_limits,
                                                                epochs=10)
        else:
            rets, us, usa, ents, sps, epls = evaluator.evaluate(epochs=10)


        returns.append(rets)
        entropies.append(ents)
        sparsities.append(sps)
        episode_lengts.append(epls)
        actions.append(buffer.action.flatten().tolist())

        results.append([env, buffer_type, (np.mean(rets), np.std(rets)), usa, (np.mean(ents), np.std(ents))])

        mm.append([env, buffer_type, (np.mean(rets), np.std(rets)), us, usa, (np.mean(ents), np.std(ents)),
                   (np.mean(sps), np.std(sps)), (np.mean(epls), np.std(epls))])

    with open(os.path.join("data", f"ex{experiment}", f"metrics_{env}_run{use_run}.pkl"), "wb") as f:
        pickle.dump(mm, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--online", action="store_true")  # run online task
    parser.add_argument("--run", default=1, type=int)  # which number of run for dataset?
    parser.add_argument("--dataset", default=-1, type=int) # which dataset to use
    args = parser.parse_args()

    assert args.dataset < 5, "dataset must be within the created ones or negative for all!"

    if args.online:
        with Pool(multiple_useruns, maxtasksperchild=1) as p:
            p.map(create_ds, range(1, multiple_useruns + 1))
            p.map(assess_env, range(1, multiple_useruns + 1))
    else:
        with Pool(multiple_useruns, maxtasksperchild=1) as p:
            p.map(train, zip(range(1, multiple_useruns + 1), [args.run] * 5, [args.dataset]*5))
