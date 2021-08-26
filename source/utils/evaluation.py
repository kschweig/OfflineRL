import numpy as np
import warnings


def evaluate(env, agent, writer, all_rewards):
    rewards, values, values_std, action_values, entropies, qval_delta = [], [], [], [], [], []

    # execute 100 episodes and average over return
    for seed in range(10):
        done, ep_reward = False, []

        env.seed(seed)
        state = env.reset()

        while not done:
            action, value, entropy = agent.policy(state, eval=True)
            state, reward, done, _ = env.step(action)
            ep_reward.append(reward)
            values.append(value.numpy().mean())
            values_std.append(value.numpy().std())
            if len(value.view(-1)) > action:
                action_values.append(value.view(-1)[action].item())
            else:
                action_values.append(np.nan)
            entropies.append(entropy)
        rewards.append(sum(ep_reward))

    all_rewards.append(np.mean(rewards))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        writer.add_scalar("eval/Reward", np.nanmean(rewards), len(all_rewards))
        writer.add_scalar("eval/Reward (SMA-10)", np.nanmean(all_rewards[-10:]), len(all_rewards))
        writer.add_scalar("eval/Max-Action-Value (mean)", np.nanmean(action_values), len(all_rewards))
        writer.add_scalar("eval/Max-Action-Value (std)", np.nanstd(action_values), len(all_rewards))
        writer.add_scalar("eval/Values", np.nanmean(values), len(all_rewards))
        writer.add_scalar("eval/Action-Values std", np.nanmean(values_std), len(all_rewards))
        writer.add_scalar("eval/Entropy", np.nanmean(entropies), len(all_rewards))

    return all_rewards


def entropy(values):
    probs = values.detach().cpu().numpy()
    # if entropy degrades
    if np.min(probs) < 1e-5:
        return 0
    return -np.sum(probs * np.log(probs))
