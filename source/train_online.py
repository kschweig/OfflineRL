import os
import torch
import pickle
import warnings
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .utils.buffer import ReplayBuffer
from .utils.evaluation import evaluate
from .utils.utils import get_agent, make_env


def train_online(experiment, agent_type="DQN", discount=0.95, envid='CartPole-v1', transitions=100000,
                 buffer_size=50000, run=1, seed=42):

    # different seed per run -> 42, 142, 242, ...
    seed += (run - 1) * 100
    # keep training parameters for online training fixed, the experiment does not interfere here.
    batch_size = 32
    lr = 1e-4
    evaluate_every = transitions // 500
    train_every = 1
    train_start_iter = batch_size

    writer = SummaryWriter(
        log_dir=os.path.join("runs", f"ex{experiment}", f"{envid}", "online", f"{agent_type}", f"run{run}"))

    env = make_env(envid)
    eval_env = make_env(envid)
    obs_space = len(env.observation_space.high)

    agent = get_agent(agent_type, obs_space, env.action_space.n, discount, lr, seed)

    # two buffers, one for learning, one for storing all transitions
    buffer = ReplayBuffer(obs_space, buffer_size, batch_size, seed=seed)
    er_buffer = ReplayBuffer(obs_space, transitions, batch_size, seed=seed)
    final_policy_buffer = ReplayBuffer(obs_space, transitions, batch_size, seed=seed)
    noisy_policy_buffer = ReplayBuffer(obs_space, transitions, batch_size, seed=seed)
    random_buffer = ReplayBuffer(obs_space, transitions, batch_size, seed=seed)

    # seeding
    env.seed(seed)
    eval_env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ep_rewards, all_rewards, all_avds = [], [], []
    done = True
    ep = 0

    #####################################
    # train agent
    #####################################
    for iteration in tqdm(range(transitions), desc=f"Behavioral policy ({envid}), run {run}"):
        # Reset if environment is done
        if done:
            if iteration > 0:
                ep_rewards.append(ep_reward)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    writer.add_scalar("train/Reward (SMA-10)", np.mean(ep_rewards[-10:]), ep)
                    writer.add_scalar("train/Reward", ep_reward, ep)
                    writer.add_scalar("train/Max-Action-Value (mean)", np.nanmean(action_values), ep)
                    writer.add_scalar("train/Max-Action-Value (std)", np.nanstd(action_values), ep)
                    writer.add_scalar("train/Values", np.nanmean(values), ep)
                    writer.add_scalar("train/Action-Values std", np.nanmean(values_std), ep)
                    writer.add_scalar("train/Entropy", np.nanmean(entropies), ep)

            state = env.reset()
            ep_reward, values, values_std, action_values, entropies = 0, [], [], [], []
            ep += 1

        # obtain action
        action, value, entropy = agent.policy(state)

        # step in environment
        next_state, reward, done, _ = env.step(action)

        # add to buffer
        buffer.add(state, action, reward, done)
        er_buffer.add(state, action, reward, done)

        # add reward, value and entropy of current step for means over episode
        ep_reward += reward
        try:
            values.append(value.numpy().mean())
            values_std.append(value.numpy().std())
            action_values.append(value.max().item())
        except AttributeError:
            values.append(value)
            values_std.append(value)
            action_values.append(value)
        entropies.append(entropy)

        # now next state is the new state
        state = next_state

        # update the agent periodically after initially populating the buffer
        if (iteration + 1) % train_every == 0 and iteration > train_start_iter:
            agent.train(buffer, writer)

        # test agent on environment if executed greedily
        if (iteration + 1) % evaluate_every == 0:
            all_rewards, all_avds = evaluate(eval_env, agent, writer, all_rewards, all_avds)

    # save ER-buffer for offline training
    os.makedirs(os.path.join("data", f"ex{experiment}"), exist_ok=True)
    with open(os.path.join("data", f"ex{experiment}", f"{envid}_run{run}_er.pkl"), "wb") as f:
        pickle.dump(er_buffer, f)
    # free memory
    del er_buffer

    # save returns of online training
    os.makedirs(os.path.join("results", "raw", "return", f"ex{experiment}"), exist_ok=True)
    with open(os.path.join("results", "raw", "return", f"ex{experiment}", f"{envid}_online_run{run}.csv"), "w") as f:
        for r in all_rewards:
            f.write(f"{r}\n")

    # save avds of online training
    os.makedirs(os.path.join("results", "raw", "avd", f"ex{experiment}"), exist_ok=True)
    with open(os.path.join("results", "raw", "avd", f"ex{experiment}", f"{envid}_online_run{run}.csv"), "w") as f:
        for avd in all_avds:
            f.write(f"{avd}\n")

    #####################################
    # generate transitions from trained agent
    #####################################
    done, n_actions = True, env.action_space.n
    for _ in tqdm(range(transitions), desc=f"Evaluate final policy ({envid}), run {run}"):
        if done:
            state = env.reset()

        action, _, _ = agent.policy(state, eval=True)

        next_state, reward, done, _ = env.step(action)

        final_policy_buffer.add(state, action, reward, done)

        state = next_state

    os.makedirs(os.path.join("data", f"ex{experiment}"), exist_ok=True)
    with open(os.path.join("data", f"ex{experiment}", f"{envid}_run{run}_fully.pkl"), "wb") as f:
        pickle.dump(final_policy_buffer, f)

    #####################################
    # generate noisy transitions from trained agent
    #####################################
    done, n_actions = True, env.action_space.n
    # make agent noisy
    agent.eval_eps = 0.2
    for _ in tqdm(range(transitions), desc=f"Evaluate noisy final policy ({envid}), run {run}"):
        if done:
            state = env.reset()

        action, _, _ = agent.policy(state, eval=True)

        next_state, reward, done, _ = env.step(action)

        noisy_policy_buffer.add(state, action, reward, done)

        state = next_state

    os.makedirs(os.path.join("data", f"ex{experiment}"), exist_ok=True)
    with open(os.path.join("data", f"ex{experiment}", f"{envid}_run{run}_noisy.pkl"), "wb") as f:
        pickle.dump(noisy_policy_buffer, f)

    #####################################
    # generate random transitions
    #####################################
    done, rng, n_actions = True, np.random.default_rng(seed), env.action_space.n
    for _ in tqdm(range(transitions), desc=f"Evaluate random policy ({envid}), run {run}"):
        if done:
            state = env.reset()

        action = rng.integers(n_actions)
        next_state, reward, done, _ = env.step(action)

        random_buffer.add(state, action, reward, done)

        state = next_state

    os.makedirs(os.path.join("data", f"ex{experiment}"), exist_ok=True)
    with open(os.path.join("data", f"ex{experiment}", f"{envid}_run{run}_random.pkl"), "wb") as f:
        pickle.dump(random_buffer, f)

    #####################################
    # generate mixed transitions (random + fully)
    #####################################
    random_buffer.mix(buffer=final_policy_buffer, p_orig=0.8)
    os.makedirs(os.path.join("data", f"ex{experiment}"), exist_ok=True)
    with open(os.path.join("data", f"ex{experiment}", f"{envid}_run{run}_mixed.pkl"), "wb") as f:
        pickle.dump(random_buffer, f)

    return agent
