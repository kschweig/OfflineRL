import os
import numpy as np
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt

plt.ioff()
import seaborn as sns

sns.set()

matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)

useruns = 5

image_type = "png"

envs = ['CartPole-v1', "MountainCar-v0", "MiniGrid-LavaGapS7-v0", "MiniGrid-Dynamic-Obstacles-8x8-v0",
        "Breakout-MinAtar-v0", "SpaceInvaders-MinAtar-v0"]
minatar = ["Breakout-MinAtar-v0", "SpaceInvaders-MinAtar-v0"]
small_minatar = True
iterations = [100000, 100000, 100000, 100000, 2000000, 2000000]
algos = ["BC", "BVE", "MCE", "DQN", "QRDQN", "REM", "BCQ", "CQL", "CRR"]
datasets = ["random", "mixed", "er", "noisy", "fully"]

y_bounds = {'CartPole-v1': (-15, 15), "MiniGrid-LavaGapS7-v0":(-0.5, 1.3), 'MountainCar-v0': (-50, 100),
            "MiniGrid-Dynamic-Obstacles-8x8-v0":(-1, 1), 'Breakout-MinAtar-v0': (-5, 25), "SpaceInvaders-MinAtar-v0": (-5, 25)}

folders = ["avd", "return"]

origin = os.path.join("..", "..", "results", "csv_per_userun")
target = os.path.join("..", "..", "results", "learning")


#### online

for e, env in enumerate(tqdm(envs)):
    for folder in folders:
        f, axs = plt.subplots(2, 3, figsize=(12, 10), sharex=True, sharey=True)
        axs = [item for sublist in zip(axs[:, 0], axs[:, 1], axs[:, 2]) for item in sublist]

        all = []
        for r in range(1, useruns + 1):
            csv = np.genfromtxt(os.path.join(origin, folder, env, f"userun{r}", "DQN_online.csv"), delimiter=";")
            axs[r].plot(np.linspace(iterations[e] / len(csv), iterations[e], len(csv)), csv, color="black")
            all.append(csv.tolist())
            axs[r].set_title(f"Run {r}")
            axs[r].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

        csv = np.asarray(all).transpose(1, 0)

        est = np.mean(csv, axis=1)
        sd = np.std(csv, axis=1)
        cis = (est - sd, est + sd)

        axs[0].fill_between(np.linspace(iterations[e]/ len(csv), iterations[e], len(csv)),
                            cis[0], cis[1], alpha=0.2, color="black")
        axs[0].plot(np.linspace(iterations[e] / len(csv), iterations[e], len(csv)),
                np.mean(csv, axis=1), color="black")
        axs[0].set_title(f"Overall")
        axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

        if folder == "avd":
            for ax in axs:
                ax.set_ylim(bottom=y_bounds[env][0], top=y_bounds[env][1])

        f.tight_layout(rect=(0.022, 0.022, 1, 0.96))
        f.text(0.53, 0.96, f"{env}", ha='center', fontsize="x-large")
        f.text(0.53, 0.01, "Update Steps", ha='center', fontsize="large")
        f.text(0.005, 0.5, "Action-Value deviation" if folder == "avd" else "Return"
               , va='center', rotation='vertical', fontsize="large")

        os.makedirs(os.path.join(target, folder, env), exist_ok=True)
        plt.savefig(os.path.join(target, folder, env, f"DQN_online." + image_type))
        plt.close()

#### per algorithm

for e, env in enumerate(envs):
    for ds in tqdm(datasets, desc=env):
        for folder in folders:
            if env in minatar and small_minatar:
                f, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
                axs = [item for sublist in zip(axs[:, 0], axs[:, 1]) for item in sublist]
                algos_ = ["BC", "DQN", "BCQ", "CQL"]
            else:
                f, axs = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)
                axs = [item for sublist in zip(axs[:, 0], axs[:, 1], axs[:, 2]) for item in sublist]
                algos_ = algos

            for a, algo in enumerate(algos_):
                for userun in range(1, useruns + 1):
                    try:
                        csv = np.genfromtxt(os.path.join(origin, folder, env, f"userun{userun}", f"{algo}_{ds}.csv"),
                                            delimiter=";")

                        if len(csv.shape) == 2:
                            est = np.mean(csv, axis=1)
                            sd = np.std(csv, axis=1)
                            cis = (est - sd, est + sd)

                            axs[a].fill_between(np.linspace(iterations[e] * 5 / len(csv), iterations[e] * 5, len(csv)),
                                                cis[0], cis[1], alpha=0.2, color=f"C{userun}")
                            csv = np.mean(csv, axis=1)
                        axs[a].plot(np.linspace(iterations[e] * 5 / len(csv), iterations[e] * 5, len(csv)),
                                    csv, color=f"C{userun}")
                        axs[a].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                    except OSError:
                        pass
                axs[a].set_title(algo)

            f.tight_layout(rect=(0.022, 0.022, 1, 0.96))
            f.text(0.53, 0.96, f"{env} / {ds}", ha='center', fontsize="x-large")
            f.text(0.53, 0.01, "Update Steps", ha='center', fontsize="large")
            f.text(0.005, 0.5, "Action-Value deviation" if folder == "avd" else "Return"
                   , va='center', rotation='vertical', fontsize="large")

            if folder == "avd":
                for ax in axs:
                    ax.set_ylim(bottom=y_bounds[env][0], top=y_bounds[env][1])

            os.makedirs(os.path.join(target, folder, env), exist_ok=True)
            plt.savefig(os.path.join(target, folder, env, f"{ds}." + image_type))
            plt.close()
