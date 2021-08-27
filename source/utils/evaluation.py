import numpy as np
import warnings


def evaluate(env, agent, writer, all_rewards, all_avds):
    rewards, entropies, qval_deltas = [], [], []

    # execute 10 episodes and average over return
    for seed in range(10):
        done, ep_reward, values, action_values = False, [], [], []

        env.seed(seed)
        state = env.reset()

        while not done:
            action, value, entropy = agent.policy(state, eval=True)
            state, reward, done, _ = env.step(action)
            ep_reward.append(reward)
            values.append(value.numpy().mean())
            if len(value.view(-1)) > action:
                action_values.append(value.view(-1)[action].item())
            else:
                action_values.append(np.nan)
            entropies.append(entropy)
        rewards.append(sum(ep_reward))

        # calculate target discounted reward
        cum_reward, cr = np.zeros_like(ep_reward), 0
        for i in reversed(range(len(ep_reward))):
            cr = cr + ep_reward[i]
            cum_reward[i] = cr
            cr *= agent.discount

        for i, qval in enumerate(action_values):
            # compare action value with real outcome
            qval_deltas.append(qval - cum_reward[i])

    all_rewards.append(np.mean(rewards))
    all_avds.append(np.mean(qval_deltas))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        writer.add_scalar("eval/Reward", np.nanmean(rewards), len(all_rewards))
        writer.add_scalar("eval/Reward (SMA-10)", np.nanmean(all_rewards[-10:]), len(all_rewards))
        writer.add_scalar("eval/Action-value deviation", np.nanmean(qval_deltas), len(all_rewards))
        writer.add_scalar("eval/Entropy", np.nanmean(entropies), len(all_rewards))

    return all_rewards, all_avds


def entropy(values):
    probs = values.detach().cpu().numpy()
    # if entropy degrades
    if np.min(probs) < 1e-5:
        return 0
    return -np.sum(probs * np.log(probs))
