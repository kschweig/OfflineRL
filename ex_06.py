from source.train_online import train_online
from source.train_offline import train_offline
from source.offline_ds_evaluation.evaluator import Evaluator
from source.offline_ds_evaluation.metrics_manager import MetricsManager
from source.offline_ds_evaluation.latex import create_latex_table
from source.offline_ds_evaluation.plotting import plot_returns, plot_actions, plot_entropies, plot_eplengths, plot_sparsities
from multiprocessing import Pool
import os
import copy
import pickle
import numpy as np


# project parameters
envs = ['CartPole-v1', "MiniGrid-LavaGapS7-v0"]
discounts = [0.95, 0.95]
buffer_types = ["random", "mixed", "er", "noisy", "fully"]
agent_types = ["BC", "BVE", "MCE", "DQN", "QRDQN", "REM", "BCQ", "CQL", "CRR"]
multiple_runs = 5
# experiment parameters
experiment = 6
seed = 42
# hyperparameters for online training
behavioral = "DQN"
transitions_online = 100000
# hyperparameters for offline training
transitions_offline = 2 * transitions_online
batch_size = 128
lr = [1e-4] * len(agent_types)


def create_ds(args):
    envid, discount = args

    train_online(experiment=experiment, agent_type=behavioral, discount=discount, envid=envid,
                 transitions=transitions_online, buffer_size=50000,
                 run=1, seed=seed)

def train(args):
    envid, discount = args

    for run in range(1, multiple_runs + 1):
        for a, agent in enumerate(agent_types):
            for bt in range(len(buffer_types)):
                train_offline(experiment=experiment, envid=envid, agent_type=agent, buffer_type=buffer_types[bt],
                              discount=discount, transitions=transitions_offline, batch_size=batch_size, lr=lr[a],
                              use_run=1, run=run, seed=seed+run, use_remaining_reward=(agent == "MCE"))

def assess_env(args):
    e, envid = args

    os.makedirs(os.path.join("results", "ds_eval"), exist_ok=True)

    with open(os.path.join("data", f"ex{experiment}", f"{envid}_run1_er.pkl"), "rb") as f:
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
        with open(os.path.join("data", f"ex{experiment}", f"{envid}_run1_{buffer_type}.pkl"), "rb") as f:
            buffer = pickle.load(f)

        evaluator = Evaluator(envid, buffer_type, buffer.state, buffer.action, buffer.reward,
                              np.invert(buffer.not_done))

        if envid == 'CartPole-v1':
            rets, us, usa, ents, sps, epls = evaluator.evaluate(state_limits=state_limits, action_limits=action_limits,
                                                                epochs=10)
        else:
            rets, us, usa, ents, sps, epls = evaluator.evaluate(epochs=10)


        returns.append(rets)
        entropies.append(ents)
        sparsities.append(sps)
        episode_lengts.append(epls)
        actions.append(buffer.action.flatten().tolist())

        results.append([envid, buffer_type, (np.mean(rets), np.std(rets)), usa, (np.mean(ents), np.std(ents))])

        mm.append([envid, buffer_type, (np.mean(rets), np.std(rets)), us, usa, (np.mean(ents), np.std(ents)),
                   (np.mean(sps), np.std(sps)), (np.mean(epls), np.std(epls))])


    create_latex_table(os.path.join("results", "ds_eval", f"{results[0][0]}.tex"), results)

    buffer = {"random": "Random Policy", "mixed": "Mixed Policy", "er": "Experience Replay",
              "noisy": "Noisy Policy", "fully": "Final Policy"}
    types = [buffer[bt] for bt in buffer_types]

    plot_returns(os.path.join("results", "ds_eval", f"{envid}_return.png"), returns, types)
    plot_actions(os.path.join("results", "ds_eval", f"{envid}_action.png"), actions, types)
    plot_entropies(os.path.join("results", "ds_eval", f"{envid}_entropy.png"), entropies, types)
    plot_eplengths(os.path.join("results", "ds_eval", f"{envid}_eplength.png"), episode_lengts, types)
    plot_sparsities(os.path.join("results", "ds_eval", f"{envid}_sparsity.png"), sparsities, types)

    with open(os.path.join("data", f"ex{experiment}", f"metrics_{envid}.pkl"), "wb") as f:
        pickle.dump(mm, f)


if __name__ == '__main__':

    with Pool(len(envs), maxtasksperchild=1) as p:
        #p.map(create_ds, zip(envs, discounts))
        #p.map(train, zip(envs, discounts))
        p.map(assess_env, zip(range(len(envs)), envs))

