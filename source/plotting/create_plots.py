import os
import glob
import scipy
import pickle
import numpy as np
from source.offline_ds_evaluation.metrics_manager import MetricsManager
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
# Turn interactive plotting off
plt.ioff()
import seaborn as sns
sns.set()

matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)


folder = "main_figures_paper"
image_type = "pdf"
"""
folder = "main_figures"
image_type = "png"
"""

figsize = (12, 6)
figsize_legend = (12, 1)
figsize_half = (12, 3.5)
figsize_half_half = (9.25, 3.5)
figsize_small_avg = (9, 3.2)
figsize_small = (13, 3.1)
figsize_comp = (12, 6)
figsize_comp_rot = (9, 8)
figsize_envs = (12, 7.2)
figsize_theplot = (13, 12)
figsize_thesmallplot = (9, 8)

# metric manager
experiments = ["ex1", "ex2", "ex3", "ex4", "ex5", "ex6"]

mm = MetricsManager(0)
useruns = 5

for ex in experiments:
    for userun in range(1, 6):
        paths = glob.glob(os.path.join("..", "..", "data", ex, f"metrics_*_run{userun}.pkl"))
        for path in paths:
            with open(path, "rb") as f:
                m = pickle.load(f)
                m.recode(userun)
            mm.data.update(m.data)

# static stuff

envs = {'CartPole-v1': 0, 'MountainCar-v0': 1, "MiniGrid-LavaGapS7-v0": 2, "MiniGrid-Dynamic-Obstacles-8x8-v0": 3,
        'Breakout-MinAtar-v0': 4, "SpaceInvaders-MinAtar-v0": 5}

algos = ["BC", "BVE", "MCE", "DQN", "QRDQN", "REM", "BCQ", "CQL", "CRR"]

buffer = {"random": "Random", "mixed": "Mixed", "er": "Replay", "noisy": "Noisy", "fully": "Expert"}

mc_actions = ["Acc. to the Left", "Don't accelerate", "Acc. to the Right"]

modes = list(buffer.keys())

markers = ["o", "D", "*", "<", "s"]

datasets = [Line2D([0], [0], color="black", marker=markers[i], linewidth=0) for i in range(len(markers))]


def plt_csv(ax, csv, algo, mode, ylims=None, set_title=True, color=None, set_label=True):
    est = np.mean(csv, axis=1)
    sd = np.std(csv, axis=1)
    cis = (est - sd, est + sd)

    ax.fill_between(np.arange(0, len(est) * 100, 100), cis[0], cis[1], alpha=0.2, color=color)
    ax.plot(np.arange(0, len(est) * 100, 100), est, label=(algo if set_label else None), color=color)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    if set_title:
        ax.set_title(buffer[mode])
    if ylims != None:
        ax.set_ylim(bottom=ylims[0], top=ylims[1])


## get data
indir = os.path.join("..", "..", "results", "csv_per_userun", "return")

files = []
for file in glob.iglob(os.path.join(indir, "**", "*.csv"), recursive=True):
    files.append(file)

data = dict()
for file in files:

    name = file.split("/")[-1]
    userun = int(file.split("/")[-2][-1])
    env = file.split("/")[-3]
    algo = name.split("_")[-2]
    mode = name.split("_")[-1].split(".")[0]


    try:
        csv = np.loadtxt(file, delimiter=";")
    except:
        print("Error in ", env, mode, algo)

    if len(csv.shape) == 1:
        csv = csv.reshape(-1, 1)

    if not data.keys() or env not in data.keys():
        data[env] = dict()
    if not data[env].keys() or userun not in data[env].keys():
        data[env][userun] = dict()
    if not data[env][userun].keys() or mode not in data[env][userun].keys():
        data[env][userun][mode] = dict()

    data[env][userun][mode][algo] = csv


###############
# plot metrics for policies
###############

outdir = os.path.join("..", "..", "results", folder, "metrics")
os.makedirs(outdir, exist_ok=True)

# titles
x_label = "Dataset"

# compact representation averaged over envs
f, axs = plt.subplots(1, 3, figsize=figsize_half, sharex=True)
for m, metric in enumerate([(0, 0), 2, (3, 0)]):
    x_all = []
    for mode in modes:
        x = []
        for e, env in enumerate(envs):
            for userun in range(1, useruns + 1):
                random_return = mm.get_data(env, "random", userun)[0][0]
                online_usap = mm.get_data(env, "er", userun)[2]

                if m == 1:
                    result = mm.get_data(env, mode, userun)[metric]
                else:
                    result = mm.get_data(env, mode, userun)[metric[0]][metric[1]]

                if m == 0:
                    csv = data[env][userun]["online"]["DQN"]
                    x.append((result - random_return) / (np.max(csv) - random_return))
                elif m == 1:
                    x.append(result / online_usap)
                else:
                    x.append(result)

        x_all.append(x)

    axs[m].boxplot(x_all, positions=range(len(modes)), widths=0.4, zorder=20,
                   medianprops={"c": f"darkcyan", "linewidth": 1.2},
                   boxprops={"c": f"darkcyan", "linewidth": 1.2},
                   whiskerprops={"c": f"darkcyan", "linewidth": 1.2},
                   capprops={"c": f"darkcyan", "linewidth": 1.2},
                   flierprops={"markeredgecolor": f"darkcyan"})#, "markeredgewidth": 1.5})

    if m == 0:
        axs[m].set_ylabel("TQ", fontsize="medium")
        axs[m].axhline(y=1, color="silver")
    elif m == 1:
        axs[m].set_ylabel("SACo", fontsize="medium")
        axs[m].axhline(y=1, color="silver")
    elif m == 2:
        axs[m].set_ylabel("Entropy", fontsize="medium")

    axs[m].set_ylim(bottom=-0.05, top=1.45)
    axs[m].set_xticks([i for i in range(len(modes))])
    axs[m].set_xticklabels([buffer[m] for m in modes],fontsize="small")#, rotation=15, rotation_mode="anchor")
axs[-1].set_ylim(bottom=-0.05, top=1.05)
f.tight_layout(rect=(0, 0.022, 1, 1))
f.text(0.52, 0.01, x_label, ha='center')
plt.savefig(os.path.join(outdir, f"overview_3_avg." + image_type))
plt.close()

# compact representation averaged over envs
f, axs = plt.subplots(1, 2, figsize=figsize_small_avg, sharex=True)
for m, metric in enumerate([(0, 0), 2]):
    x_all = []
    for mode in modes:
        x = []
        for e, env in enumerate(envs):
            for userun in range(1, useruns + 1):
                random_return = mm.get_data(env, "random", userun)[0][0]
                online_usap = mm.get_data(env, "er", userun)[2]

                if m == 1:
                    result = mm.get_data(env, mode, userun)[metric]
                else:
                    result = mm.get_data(env, mode, userun)[metric[0]][metric[1]]

                if m == 0:
                    csv = data[env][userun]["online"]["DQN"]
                    x.append((result - random_return) / (np.max(csv) - random_return))
                elif m == 1:
                    x.append(result / online_usap)

        x_all.append(x)

    axs[m].boxplot(x_all, positions=range(len(modes)), widths=0.4, zorder=20,
                   medianprops={"c": "darkcyan", "linewidth": 1.2},
                   boxprops={"c": "darkcyan", "linewidth": 1.2},
                   whiskerprops={"c": "darkcyan", "linewidth": 1.2},
                   capprops={"c": "darkcyan", "linewidth": 1.2},
                   flierprops={"markeredgecolor": "darkcyan"})#, "markeredgewidth": 1.5})
    axs[m].scatter(range(len(modes)), [np.mean(x_) for x_ in x_all], color="indianred")

    if m == 0:
        axs[m].set_ylabel("TQ", fontsize="medium")
        axs[m].axhline(y=1, color="silver")
    elif m == 1:
        axs[m].set_ylabel("SACo", fontsize="medium")
        axs[m].axhline(y=1, color="silver")

    axs[m].set_ylim(bottom=-0.05, top=1.45)
    axs[m].set_xticks([i for i in range(len(modes))])
    axs[m].set_xticklabels([buffer[m] for m in modes],fontsize="small")#, rotation=15, rotation_mode="anchor")

f.tight_layout(rect=(0, 0.022, 1, 1))
f.text(0.52, 0.01, x_label, ha='center', fontsize="medium")
plt.savefig(os.path.join(outdir, f"overview_2_avg." + image_type))
plt.close()


# compact representation
f, axs = plt.subplots(1, 3, figsize=figsize_half, sharex=True)
for m, metric in enumerate([(0, 0), 2, (3, 0)]):
    for e, env in enumerate(envs):
        x_all = []
        for mode in modes:
            x = []
            for userun in range(1, useruns + 1):
                random_return = mm.get_data(env, "random", userun)[0][0]
                online_usap = mm.get_data(env, "er", userun)[2]

                if m == 1:
                    result = mm.get_data(env, mode, userun)[metric]
                else:
                    result = mm.get_data(env, mode, userun)[metric[0]][metric[1]]

                if m == 0:
                    csv = data[env][userun]["online"]["DQN"]
                    x.append((result - random_return) / (np.max(csv) - random_return))
                elif m == 1:
                    x.append(result / online_usap)
                else:
                    x.append(result)
            x_all.append(x)

        pos = [0.2 + 0.12 * e + m_ for m_ in range(len(modes))]

        axs[m].boxplot(x_all, positions=pos, widths=0.1, sym="", zorder=20,
                       medianprops={"c": f"C{e}", "linewidth": 1.5},
                       boxprops={"color": f"C{e}", "linewidth": 1.5},
                       whiskerprops={"color": f"C{e}", "linewidth": 1.5},
                       capprops={"color": f"C{e}", "linewidth": 1.5},
                       flierprops={"color": f"C{e}", "linewidth": 1.5})

    if m == 0:
        axs[m].set_ylabel("TQ", fontsize="medium")
        axs[m].axhline(y=1, color="silver")
    elif m == 1:
        axs[m].set_ylabel("SACo", fontsize="medium")
        axs[m].axhline(y=1, color="silver")
    elif m == 2:
        axs[m].set_ylabel("Entropy", fontsize="medium")

    axs[m].set_ylim(bottom=-0.05, top=1.45)
    axs[m].set_xticks([i for i in range(len(modes) + 1)])
    names = [buffer[m] for m in modes]
    names.append("")
    axs[m].set_xticklabels(names,fontsize="small")#, rotation=15, rotation_mode="anchor")
    offset = matplotlib.transforms.ScaledTranslation(0.29, 0, f.dpi_scale_trans)
    for label in axs[m].xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
axs[-1].set_ylim(bottom=-0.05, top=1.05)
labels = [mpatches.Patch(color=f"C{e_}", fill=False, linewidth=1.5, label="-".join(env_.split("-")[:-1])) for e_, env_ in enumerate(envs)]
f.legend(handles=labels, handlelength=1, loc="upper center", ncol=len(envs), fontsize="small")
f.tight_layout(rect=(0, 0.022, 1, 0.92))
f.text(0.52, 0.01, x_label, ha='center')
plt.savefig(os.path.join(outdir, f"overview_3." + image_type))
plt.close()


# compact representation
f, axs = plt.subplots(1, 2, figsize=figsize_half_half, sharex=True)
for m, metric in enumerate([(0, 0), 2]):
    for e, env in enumerate(envs):
        x_all = []
        for mode in modes:
            x = []
            for userun in range(1, useruns + 1):
                random_return = mm.get_data(env, "random", userun)[0][0]
                online_usap = mm.get_data(env, "er", userun)[2]

                if m == 1:
                    result = mm.get_data(env, mode, userun)[metric]
                else:
                    result = mm.get_data(env, mode, userun)[metric[0]][metric[1]]

                if m == 0:
                    csv = data[env][userun]["online"]["DQN"]
                    x.append((result - random_return) / (np.max(csv) - random_return))
                elif m == 1:
                    x.append(result / online_usap)
                else:
                    x.append(result)
            x_all.append(x)

        pos = [0.2 + 0.12 * e + m_ for m_ in range(len(modes))]

        axs[m].boxplot(x_all, positions=pos, widths=0.1, sym="", zorder=20,
                       medianprops={"c": f"C{e}", "linewidth": 1.5},
                       boxprops={"color": f"C{e}", "linewidth": 1.5},
                       whiskerprops={"color": f"C{e}", "linewidth": 1.5},
                       capprops={"color": f"C{e}", "linewidth": 1.5},
                       flierprops={"color": f"C{e}", "linewidth": 1.5})
        #axs[m].plot(range(len(x)), x, "-o", label = "-".join(env.split("-")[:-1]) if m == 0 else None, zorder=20)

    if m == 0:
        axs[m].set_ylabel("TQ", fontsize="medium")
        axs[m].axhline(y=1, color="silver")
    elif m == 1:
        axs[m].set_ylabel("SACo", fontsize="medium")
        axs[m].axhline(y=1, color="silver")

    axs[m].set_ylim(bottom=-0.05, top=1.45)
    axs[m].set_xticks([i for i in range(len(modes) + 1)])
    names = [buffer[m] for m in modes]
    names.append("")
    axs[m].set_xticklabels(names, fontsize="small")#, rotation=15, rotation_mode="anchor")
    offset = matplotlib.transforms.ScaledTranslation(0.33, 0, f.dpi_scale_trans)
    for label in axs[m].xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)

labels = [mpatches.Patch(color=f"C{e_}", fill=False, linewidth=1.5, label="-".join(env_.split("-")[:-1])) for e_, env_ in enumerate(envs)]
f.legend(handles=labels, handlelength=1, loc="upper center", ncol=len(envs), fontsize="x-small")
f.tight_layout(rect=(0, 0.022, 1, 0.92))
f.text(0.52, 0.01, x_label, ha='center')
plt.savefig(os.path.join(outdir, f"overview_2." + image_type))
plt.close()


###############
# Main Results
###############

outdir = os.path.join("..", "..", "results", folder, "tq_vs_sac")
os.makedirs(outdir, exist_ok=True)

from  matplotlib.colors import LinearSegmentedColormap
#c = ["seagreen", "darkcyan", ""]#["red", "tomato", "lightsalmon", "wheat", "palegreen", "limegreen", "green"]
#v = [i / (len(c) - 1) for i in range(len(c))]
#print(v)
#l = list(zip(v, c))
cmap = "viridis"  #LinearSegmentedColormap.from_list('grnylw',l, N=256)
normalize = Normalize(vmin=0, vmax=120, clip=True)

offset_ann = 0.025

# titles
x_label = r"SACo of Dataset"
y_label = r"TQ of Dataset"

# plot for discussion
### algos not averaged

types = ["all", "noMinAtar", "MinAtar"]

for l, log in enumerate(["", "_log"]):
    for t, environments in enumerate([list(envs)]):

        if t == 2:
            f, axs = plt.subplots(2, 2, figsize=(figsize_thesmallplot[0], figsize_thesmallplot[1]), sharex=True, sharey=True)
            axs = [item for sublist in zip(axs[:, 0], axs[:, 1]) for item in sublist]
        else:
            f, axs = plt.subplots(3, 3, figsize=(figsize_theplot[0], figsize_theplot[1]), sharex=True, sharey=True)
            axs = [item for sublist in zip(axs[:, 0], axs[:, 1], axs[:, 2]) for item in sublist]

        for a, algo in enumerate(algos):
            ax = axs[a]

            ax.axhline(y=1, color="silver")
            ax.axvline(x=1, color="silver")
            ax.set_title(algo, fontsize="large")

            for e, env in enumerate(list(environments)):
                for userun in range(1, useruns + 1):

                    online_return = np.max(data[env][userun]["online"]["DQN"])
                    random_return = mm.get_data(env, "random", userun)[0][0]
                    online_usap = mm.get_data(env, "er", userun)[2]

                    for m, mode in enumerate(modes):
                        try:
                            performance = (np.max(np.mean(data[env][userun][mode][algo], axis=1)) - random_return) / (
                                        online_return - random_return) * 100
                            if log == "_log":
                                x = np.log(mm.get_data(env, mode, userun)[2]) / np.log(online_usap)
                            else:
                                x = mm.get_data(env, mode, userun)[2] / online_usap
                            y = (mm.get_data(env, mode, userun)[0][0] - random_return) / (online_return - random_return)
                        except:
                            continue

                        ax.scatter(x, y, s = 70, c=performance, cmap=cmap, norm=normalize, zorder=10, marker=markers[m])

            """
            for i in range(len(performance)):
                ax.annotate(f"{int(performance[i])}%", (x[i] + offset_ann, y[i] + offset_ann), fontsize="x-small", zorder=20)
            
            if a == 0:
                print("-" * 30)
                print(types[t])
                print("(TQ - SAC):", " ".join([f"{round(i, 3)}" for i in scipy.stats.pearsonr(x, y)]))
                print("-" * 30)
            print(algo, " (TQ - P):", " ".join([f"{round(i, 3)}" for i in scipy.stats.pearsonr(y, performance)]))
            print(algo, " (SAC - P):", " ".join([f"{round(i, 3)}" for i in scipy.stats.pearsonr(x, performance)]))
            print("-" * 30)
        """
        f.colorbar(matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap), ax=axs, anchor=(1.3, 0.55),
                   shrink=0.5 if t < 2 else 0.5).set_label(label="Performance in % of Online Policy", size=14)

        f.tight_layout(rect=(0.022, 0.022, 0.91, 1))
        f.legend(datasets, [buffer[m] for m in modes], loc="upper right", bbox_to_anchor=(1, 0.971))
        f.text(0.5, 0.01, x_label if log == "" else "lSACo of Dataset", ha='center', fontsize="large")
        f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
        plt.savefig(os.path.join(outdir, f"algos{log}_{types[t]}." + image_type))
        plt.close()

### algos averaged
for l, log in enumerate(["", "_log"]):
    for t, environments in enumerate([list(envs)]):
        if t == 2:
            f, axs = plt.subplots(2, 2, figsize=(figsize_thesmallplot[0], figsize_thesmallplot[1]), sharex=True,
                                  sharey=True)
            axs = [item for sublist in zip(axs[:, 0], axs[:, 1]) for item in sublist]
        else:
            f, axs = plt.subplots(3, 3, figsize=(figsize_theplot[0], figsize_theplot[1]), sharex=True, sharey=True)
            axs = [item for sublist in zip(axs[:, 0], axs[:, 1], axs[:, 2]) for item in sublist]

        for a, algo in enumerate(algos):
            ax = axs[a]

            ax.axhline(y=1, color="silver")
            ax.axvline(x=1, color="silver")
            ax.set_title(algo, fontsize="large")

            for e, env in enumerate(list(environments)):
                for m, mode in enumerate(modes):
                    x, y, performance = [], [], []
                    for userun in range(1, useruns + 1):
                        online_return = np.max(data[env][userun]["online"]["DQN"])
                        random_return = mm.get_data(env, "random", userun)[0][0]
                        online_usap = mm.get_data(env, "er", userun)[2]
                        try:
                            performance.append((np.max(np.mean(data[env][userun][mode][algo], axis=1)) - random_return) / (
                                        online_return - random_return) * 100)
                            if log == "_log":
                                x = np.log(mm.get_data(env, mode, userun)[2]) / np.log(online_usap)
                            else:
                                x = mm.get_data(env, mode, userun)[2] / online_usap
                            y.append((mm.get_data(env, mode, userun)[0][0] - random_return) / (online_return - random_return))
                        except:
                            continue

                    ax.scatter(np.mean(x), np.mean(y), s = 140, c=np.mean(performance), cmap=cmap, norm=normalize, zorder=10, marker=markers[m])

            """
            for i in range(len(performance_)):
                ax.annotate(f"{int(performance_[i])}%", (x_[i] + offset_ann, y_[i] + offset_ann), fontsize="x-small", zorder=20)
            
    
            if a == 0:
                print("-" * 30)
                print(types[t])
                print("(TQ - SAC):", " ".join([f"{round(i, 3)}" for i in scipy.stats.pearsonr(x_, y_)]))
                print("-" * 30)
            print(algo, " (TQ - P):", " ".join([f"{round(i, 3)}" for i in scipy.stats.pearsonr(y_, performance_)]))
            print(algo, " (SAC - P):", " ".join([f"{round(i, 3)}" for i in scipy.stats.pearsonr(x_, performance_)]))
            print("-" * 30)
        """
        f.colorbar(matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap), ax=axs, anchor=(1.3, 0.55),
                   shrink=0.5 if t < 2 else 0.5).set_label(label="Performance in % of Online Policy", size=14)

        f.tight_layout(rect=(0.022, 0.022, 0.91, 1))
        f.legend(datasets, [buffer[m] for m in modes], loc="upper right", bbox_to_anchor=(1, 0.971))
        f.text(0.5, 0.01, x_label if log == "" else "lSACo of Dataset", ha='center', fontsize="large")
        f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
        plt.savefig(os.path.join(outdir, f"algos{log}_avg_{types[t]}." + image_type))
        plt.close()

### for environments

for method in ["Mean", "Maximum", "Minimum", "Median", "Mean + STD", "Mean - STD"]:

    f, axs = plt.subplots(2, 3, figsize=figsize_envs, sharex=True, sharey=True)
    axs = [item for sublist in zip(axs[0], axs[1]) for item in sublist]

    for e, env in enumerate(envs):
        ax = axs[e]

        ax.axhline(y=1, color="silver")
        ax.axvline(x=1, color="silver")

        for userun in range(1, useruns + 1):
            online_return = np.max(data[env][userun]["online"]["DQN"])
            random_return = mm.get_data(env, "random", userun)[0][0]
            online_usap = mm.get_data(env, "er", userun)[2]

            x, y, performance = [], [], []
            for m, mode in enumerate(modes):
                x.append(mm.get_data(env, mode, userun)[2] / online_usap)
                y.append((mm.get_data(env, mode, userun)[0][0] - random_return) / (online_return - random_return))

                p = []
                for algo in algos:
                    ax.set_title("-".join(env.split("-")[:-1]), fontsize="large")
                    try:
                        p.append((np.max(np.mean(data[env][userun][mode][algo], axis=1)) - random_return) /
                                 (online_return - random_return) * 100)
                    except:
                        pass
                performance.append(p)

            if method == "Mean":
                performance = np.mean(np.asarray(performance), axis=1)
            elif method == "Maximum":
                performance = np.max(np.asarray(performance), axis=1)
            elif method == "Minimum":
                performance = np.min(np.asarray(performance), axis=1)
            elif method == "Mean + STD":
                performance = np.mean(np.asarray(performance), axis=1) + np.std(np.asarray(performance), axis=1)
            elif method == "Mean - STD":
                performance = np.mean(np.asarray(performance), axis=1) - np.std(np.asarray(performance), axis=1)
            elif method == "Median":
                performance = np.median(np.asarray(performance), axis=1)

            ax.scatter(x, y, s=100, c=performance, cmap=cmap, norm=normalize, zorder=10)

        """
        for i in range(len(performance)):
            ax.annotate(f"{int(performance[i])}%", (x[i] + offset_ann, y[i] + offset_ann), fontsize="x-small", va="bottom", ha="left",zorder=20)
            ax.annotate(annotations[i], (x[i] - offset_ann, y[i] + offset_ann), fontsize="x-small", va="bottom", ha="right", zorder=30)
        """

    f.colorbar(matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap), ax=axs, anchor=(1.35, 0.55),
               shrink=0.5 if t < 2 else 0.5).set_label(label="Performance in % of Online Policy", size=14)

    f.tight_layout(rect=(0.022, 0.022, 0.92, 0.96))
    f.text(0.5, 0.96, f"{method} performance across algorithms", ha='center', fontsize="x-large")
    f.text(0.5, 0.01, x_label, ha='center', fontsize="large")
    f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
    plt.savefig(os.path.join(outdir, f"envs_{method}." + image_type))
    plt.close()

#### for Environments average

for method in ["Mean", "Maximum", "Minimum", "Median", "Mean + STD", "Mean - STD"]:

    f, axs = plt.subplots(2, 3, figsize=figsize_envs, sharex=True, sharey=True)
    axs = [item for sublist in zip(axs[0], axs[1]) for item in sublist]

    for e, env in enumerate(envs):
        ax = axs[e]

        ax.axhline(y=1, color="silver")
        ax.axvline(x=1, color="silver")

        for userun in range(1, useruns + 1):
            online_return = np.max(data[env][userun]["online"]["DQN"])
            random_return = mm.get_data(env, "random", userun)[0][0]
            online_usap = mm.get_data(env, "er", userun)[2]

            x, y, performance = [], [], []
            for m, mode in enumerate(modes):
                x.append(mm.get_data(env, mode, userun)[2] / online_usap)
                y.append((mm.get_data(env, mode, userun)[0][0] - random_return) / (online_return - random_return))

                p = []
                for algo in algos:
                    ax.set_title("-".join(env.split("-")[:-1]), fontsize="large")
                    try:
                        p.append((np.max(np.mean(data[env][userun][mode][algo], axis=1)) - random_return) /
                                 (online_return - random_return) * 100)
                    except:
                        pass
                performance.append(p)

        if method == "Mean":
            performance = np.mean(np.asarray(performance), axis=1)
        elif method == "Maximum":
            performance = np.max(np.asarray(performance), axis=1)
        elif method == "Minimum":
            performance = np.min(np.asarray(performance), axis=1)
        elif method == "Mean + STD":
            performance = np.mean(np.asarray(performance), axis=1) + np.std(np.asarray(performance), axis=1)
        elif method == "Mean - STD":
            performance = np.mean(np.asarray(performance), axis=1) - np.std(np.asarray(performance), axis=1)
        elif method == "Median":
            performance = np.median(np.asarray(performance), axis=1)

        ax.scatter(x, y, s=100, c=performance, cmap=cmap, norm=normalize, zorder=10)

        """
        for i in range(len(performance)):
            ax.annotate(f"{int(performance[i])}%", (x[i] + offset_ann, y[i] + offset_ann), fontsize="x-small", va="bottom", ha="left",zorder=20)
            ax.annotate(annotations[i], (x[i] - offset_ann, y[i] + offset_ann), fontsize="x-small", va="bottom", ha="right", zorder=30)
        """

    f.colorbar(matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap), ax=axs, anchor=(1.35, 0.55),
               shrink=0.5 if t < 2 else 0.5).set_label(label="Performance in % of Online Policy", size=14)

    f.tight_layout(rect=(0.022, 0.022, 0.92, 0.96))
    f.text(0.5, 0.96, f"{method} performance across algorithms", ha='center', fontsize="x-large")
    f.text(0.5, 0.01, x_label, ha='center', fontsize="large")
    f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
    plt.savefig(os.path.join(outdir, f"envs_avg_{method}." + image_type))
    plt.close()


###################################
#        Correlation plots        #
###################################

outdir = os.path.join("..", "..", "results", folder, "correlations")
os.makedirs(outdir, exist_ok=True)

y_label = r"$\bf{Performance}$ of algorithm compared to Online Policy"
x_labels = [r"TQ of Dataset",
            r"SACo of Dataset",
            r"$\bf{Entropy}$ of Dataset creating Policies"]

for pt, plot_type in enumerate(["AP_TQ", "AP_SACo", "AP_Entropy"]):
    f, axs = plt.subplots(3, 3, figsize=(figsize_theplot[0], figsize_theplot[1]), sharex=True, sharey=True)
    axs = [item for sublist in zip(axs[:, 0], axs[:, 1], axs[:, 2]) for item in sublist]
    algos_ = algos

    for a, algo in enumerate(algos_):
        ax = axs[a]

        ax.set_title(algo, fontsize="large")

        tq, saco, entropy, performance = [], [], [], []

        for e, env in enumerate(list(envs)):
            for userun in range(1, useruns + 1):

                online_return = np.max(data[env][userun]["online"]["DQN"])
                random_return = mm.get_data(env, "random", userun)[0][0]
                online_usap = mm.get_data(env, "er", userun)[2]

                for m, mode in enumerate(modes):
                    try:
                        performance.append((np.max(np.mean(data[env][userun][mode][algo], axis=1)) - random_return) / (
                                    online_return - random_return))
                        saco.append(mm.get_data(env, mode, userun)[2] / online_usap)
                        tq.append((mm.get_data(env, mode, userun)[0][0] - random_return) / (online_return - random_return))
                        entropy.append(mm.get_data(env, mode, userun)[3][0])
                    except:
                        continue

        if pt == 0:
            ax.scatter(tq, performance, s = 70, zorder=10, alpha=0.5)
            corr, pvalue = scipy.stats.pearsonr(tq, performance)
            pvalue = "{:.1e}".format(pvalue)
            ax.annotate(f"{round(corr, 2)}\n({pvalue})", (0.95, 0.95), xycoords='axes fraction', fontsize="small", va="top", ha="right", zorder=20)
        elif pt == 1:
            ax.scatter(saco, performance, s=70, zorder=10, alpha=0.5)
            corr, pvalue = scipy.stats.pearsonr(saco, performance)
            pvalue = "{:.1e}".format(pvalue)
            ax.annotate(f"{round(corr, 2)}\n({pvalue})", (0.95, 0.95), xycoords='axes fraction', fontsize="small",
                        va="top", ha="right", zorder=20)
        elif pt == 2:
            ax.scatter(entropy, performance, s=70, zorder=10, alpha=0.5)
            corr, pvalue = scipy.stats.pearsonr(entropy, performance)
            pvalue = "{:.1e}".format(pvalue)
            ax.annotate(f"{round(corr, 2)}\n({pvalue})", (0.95, 0.95), xycoords='axes fraction', fontsize="small",
                        va="top", ha="right", zorder=20)

    f.tight_layout(rect=(0.022, 0.022, 0.978, 1))
    f.text(0.5, 0.01, x_labels[pt], ha='center', fontsize="large")
    f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
    plt.savefig(os.path.join(outdir, f"{plot_type}." + image_type))
    plt.close()


##### Connection between things

f, axs = plt.subplots(1, 3, figsize=(12, 3.7))

tq, saco, entropy = [], [], []

for e, env in enumerate(list(envs)):
    for userun in range(1, useruns + 1):

        online_return = np.max(data[env][userun]["online"]["DQN"])
        random_return = mm.get_data(env, "random", userun)[0][0]
        online_usap = mm.get_data(env, "er", userun)[2]

        for m, mode in enumerate(modes):
            try:
                saco.append(mm.get_data(env, mode, userun)[2] / online_usap)
                tq.append((mm.get_data(env, mode, userun)[0][0] - random_return) / (online_return - random_return))
                entropy.append(mm.get_data(env, mode, userun)[3][0])
            except:
                continue

for pt, plot_type in enumerate(["TQ - SACo", "TQ - Entropy", "SACo - Entropy"]):
    ax = axs[pt]
    if pt == 0:
        ax.scatter(saco, tq, s=70, zorder=10, alpha=0.5)
        corr, pvalue = scipy.stats.pearsonr(saco, tq)
        pvalue = "{:.1e}".format(pvalue)
        ax.annotate(f"{round(corr, 2)}\n({pvalue})", (0.95, 0.95), xycoords='axes fraction', fontsize="small",
                    va="top", ha="right", zorder=20)
        ax.set_xlabel(x_labels[1], fontsize="small")
        ax.set_ylabel(x_labels[0], fontsize="small")
    elif pt == 1:
        ax.scatter(entropy, tq, s=70, zorder=10, alpha=0.5)
        corr, pvalue = scipy.stats.pearsonr(entropy, tq)
        pvalue = "{:.1e}".format(pvalue)
        ax.annotate(f"{round(corr, 2)}\n({pvalue})", (0.95, 0.95), xycoords='axes fraction', fontsize="small",
                    va="top", ha="right", zorder=20)
        ax.set_xlabel(x_labels[2], fontsize="small")
        ax.set_ylabel(x_labels[0], fontsize="small")
    elif pt == 2:
        ax.scatter(entropy, saco, s=70, zorder=10, alpha=0.5)
        corr, pvalue = scipy.stats.pearsonr(entropy, saco)
        pvalue = "{:.1e}".format(pvalue)
        ax.annotate(f"{round(corr, 2)}\n({pvalue})", (0.05, 0.95), xycoords='axes fraction', fontsize="small",
                    va="top", ha="left", zorder=20)
        ax.set_xlabel(x_labels[2], fontsize="small")
        ax.set_ylabel(x_labels[1], fontsize="small")

f.tight_layout()
plt.savefig(os.path.join(outdir, f"tq_saco_entropy." + image_type))
plt.close()


#############################
#        Comparisons        #
#############################

##################
# load reward data
##################
outdir = os.path.join("..", "..", "results", folder, "comp_return")
os.makedirs(outdir, exist_ok=True)

###############
# plot metrics + policy for reward
###############

# titles
y_label = "Maximum Average Return"
x_label = "Dataset"

### buffertypes per userun

for userun in range(1, useruns + 1):
    # plot for modes
    f, axs = plt.subplots(2, 3, figsize=figsize_comp, sharex=True)
    axs = [item for sublist in zip(axs[0], axs[1]) for item in sublist]

    for e, env in enumerate(envs):

        ax = axs[e]
        ax.set_title(env)

        x, y = list(range(len(buffer))), []
        for mode in modes:
            y.append(mm.get_data(env, mode, userun)[0][0])
        x, y = [list(tuple) for tuple in zip(*sorted(zip(x, y)))]

        ax.plot(x, y, "o:", label=("Behav." if e == 0 else None), color="black")

        # Online Policy
        csv = data[env][userun]["online"]["DQN"]
        ax.axhline(y=np.max(csv), color="black", label=("Online" if e == 0 else None))

        for a, algo in enumerate(algos):

            x, y, sd = [], [], []
            for m, mode in enumerate(modes):
                try:
                    y.append(np.mean(data[env][userun][mode][algo]))
                    sd.append(np.std(data[env][userun][mode][algo]))
                    x.append(m)
                except:
                    # print(env, userun, mode, algo)
                    pass

            if len(x) == 0 or len(y) == 0 or len(sd) == 0:
                continue

            x, y, sd = [list(tuple) for tuple in zip(*sorted(zip(x, y, sd)))]

            cis = (np.asarray(y) - np.asarray(sd), np.asarray(y) + np.asarray(sd))
            ax.fill_between(x, cis[0], cis[1], alpha=0.2, color=f"C{a}")
            ax.plot(x, y, "o-", label=(algo if e == 0 else None), color=f"C{a}")

        x = []
        for m, mode in enumerate(modes):
            x.append(m)

        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels([buffer[m] for m in modes], fontsize="small")#, rotation=15, rotation_mode="anchor")

    f.legend(loc="upper center", ncol=len(algos) + 2, fontsize="small")
    f.tight_layout(rect=(0.008, 0.022, 1, 0.95))
    f.text(0.52, 0.01, x_label, ha='center', fontsize="large")
    f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
    plt.savefig(os.path.join(outdir, f"buffertypes_userun{userun}." + image_type))
    plt.close()


### barplots for buffertypes

# plot for modes
f, axs = plt.subplots(3, 2, figsize=figsize_comp_rot, sharex=True)
axs = [item for sublist in zip(axs[:, 0], axs[:, 1]) for item in sublist]

width = 0.08

handles = [Patch(facecolor="black")] + [Patch(facecolor=f"C{i}") for i in range(len(algos))]

for e, env in enumerate(envs):

    ax = axs[e]
    ax.set_title(env, fontsize="small")

    # behavioral policy creating the buffer
    for m, mode in enumerate(modes):
        y = []
        for userun in range(1, useruns + 1):
            y.append(mm.get_data(env, mode, userun)[0][0])

        ax.bar(m - 4.5 * width, np.mean(y), yerr=np.std(y), color="black", width=width, error_kw=dict(lw=0.2))

    # algorithms
    for a, algo in enumerate(algos):

        x, y, sd = [], [], []
        for m, mode in enumerate(modes):
            try:
                y_ = []
                for userun in range(1, useruns + 1):
                    y_.append(np.max(np.mean(data[env][userun][mode][algo], axis=1)))
                y.append(np.mean(y_))
                sd.append(np.std(y_))
                x.append(m)
            except:
                # print(env, userun, mode, algo)
                pass

        if len(x) == 0 or len(y) == 0 or len(sd) == 0:
            continue

        x, y, sd = [list(tuple) for tuple in zip(*sorted(zip(x, y, sd)))]

        ax.bar([x_ + (a - 3.5) * width for x_ in x], y, yerr=sd, color=f"C{a}", width=width, error_kw=dict(lw=0.2))

    ax.set_xticks(np.arange(len(modes)))
    ax.set_xticklabels([buffer[m] for m in modes], fontsize="small")#, rotation=15, rotation_mode="anchor")

f.legend(handles, ["Behav."] + algos, loc="upper center", ncol=len(algos) + 1, fontsize="x-small")
f.tight_layout(rect=(0.008, 0.022, 1, 0.97))
f.text(0.52, 0.01, x_label, ha='center', fontsize="medium")
f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="medium")
plt.savefig(os.path.join(outdir, f"buffertypes." + image_type))
plt.close()

# plot for modes
f, axs = plt.subplots(3, 2, figsize=figsize_comp_rot, sharex=True)
axs = [item for sublist in zip(axs[:, 0], axs[:, 1]) for item in sublist]

width = 0.08

handles = [Patch(facecolor="black")] + [Patch(facecolor=f"C{i}") for i in range(len(algos))]

for e, env in enumerate(envs):

    ax = axs[e]
    ax.set_title(env, fontsize="small")

    # behavioral policy creating the buffer
    for m, mode in enumerate(modes):
        y = []
        for userun in range(1, useruns + 1):
            online_return = np.max(data[env][userun]["online"]["DQN"])
            random_return = mm.get_data(env, "random", userun)[0][0]
            y.append((mm.get_data(env, mode, userun)[0][0] - random_return) / (online_return - random_return))

        ax.bar(m - 4.5 * width, np.mean(y), yerr=np.std(y), color="black", width=width, error_kw=dict(lw=0.2))

    # algorithms
    for a, algo in enumerate(algos):

        x, y, sd = [], [], []
        for m, mode in enumerate(modes):
            try:
                y_ = []
                for userun in range(1, useruns + 1):
                    online_return = np.max(data[env][userun]["online"]["DQN"])
                    random_return = mm.get_data(env, "random", userun)[0][0]
                    y_.append((np.max(np.mean(data[env][userun][mode][algo], axis=1)) - random_return) / (online_return - random_return))
                y.append(np.mean(y_))
                sd.append(np.std(y_))
                x.append(m)
            except:
                # print(env, userun, mode, algo)
                pass

        if len(x) == 0 or len(y) == 0 or len(sd) == 0:
            continue

        x, y, sd = [list(tuple) for tuple in zip(*sorted(zip(x, y, sd)))]

        ax.bar([x_ + (a - 3.5) * width for x_ in x], y, yerr=sd, color=f"C{a}", width=width, error_kw=dict(lw=0.2))

    ax.set_xticks(np.arange(len(modes)))
    ax.set_xticklabels([buffer[m] for m in modes], fontsize="small")#, rotation=15, rotation_mode="anchor")

f.legend(handles, ["Behav."] + algos, loc="upper center", ncol=len(algos) + 1, fontsize="x-small")
f.tight_layout(rect=(0.008, 0.022, 1, 0.97))
f.text(0.52, 0.01, x_label, ha='center', fontsize="medium")
f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="medium")
plt.savefig(os.path.join(outdir, f"buffertypes_tq." + image_type))
plt.close()


### MountainCar
outdir = os.path.join("..", "..", "results", folder, "mountainCar")
os.makedirs(outdir, exist_ok=True)

colors=["#39568CFF", "#ffcf20FF", "#29AF7FFF"]

samples = 10000

np.random.seed(42)
ind = np.random.choice(10**5, (samples, ), replace=False)

f, axs = plt.subplots(5, 5, figsize=figsize_theplot, sharex=True, sharey=True)
axs = [item for sublist in zip(axs[0], axs[1], axs[2], axs[3], axs[4]) for item in sublist]

for m, bt in enumerate(buffer):
    for userun in range(1, useruns + 1):
        ax = axs[m*5 + userun - 1]
        # load saved buffer
        with open(os.path.join("..", "..", "data", f"ex2", f"MountainCar-v0_run{userun}_{bt}.pkl"), "rb") as file:
            er_buffer = pickle.load(file)
            if userun == 1:
                ax.set_title(buffer[bt])
            ax.scatter(er_buffer.state[ind, 0], er_buffer.state[ind, 1], c=[colors[a] for a in er_buffer.action[ind, 0]], s=0.5)
            ax.text(0.02, 0.92, f"Run {userun}", fontsize="small", transform=ax.transAxes)

f.tight_layout(rect=(0.022, 0.022, 1, 0.97))
labels = [mpatches.Patch(color=colors[a], fill=True, linewidth=1, label=mc_actions[a]) for a in range(3)]
f.legend(handles=labels, handlelength=1, loc="upper right", ncol=3, fontsize="small")
f.text(0.53, 0.98, "Dataset", ha='center', fontsize="large")
f.text(0.53, 0.01, "Position in m", ha='center', fontsize="large")
f.text(0.005, 0.5, "Velocity in m/s", va='center', rotation='vertical', fontsize="large")
plt.savefig(os.path.join(outdir, f"mountaincar.png"))
plt.close()

#### for presentation
# mountaincar dataset plot

f, axs = plt.subplots(1, 5, figsize=figsize_small, sharex=True, sharey=True)
userun = 2

for m, bt in enumerate(buffer):
    ax = axs[m]
    # load saved buffer
    with open(os.path.join("..", "..", "data", f"ex2", f"MountainCar-v0_run{userun}_{bt}.pkl"), "rb") as file:
        er_buffer = pickle.load(file)
        if userun == 2:
            ax.set_title(buffer[bt] + " Dataset")
        ax.scatter(er_buffer.state[ind, 0], er_buffer.state[ind, 1], c=[colors[a] for a in er_buffer.action[ind, 0]], s=0.5)

f.tight_layout(rect=(0.022, 0.08, 1, 1))
labels = [mpatches.Patch(color=colors[a], fill=True, linewidth=1, label=mc_actions[a]) for a in range(3)]
f.legend(handles=labels, handlelength=1, loc="lower right", ncol=3, fontsize="small")
#f.text(0.528, 0.9, "Dataset", ha='center', fontsize="large")
f.text(0.53, 0.04, "Position in m", ha='center', fontsize="large")
f.text(0.005, 0.5, "Velocity in m/s", va='center', rotation='vertical', fontsize="large")
plt.savefig(os.path.join(outdir, f"mountaincar_small.png"))
plt.close()

# mountaincar tq vs saco plot

x_label = r"SACo of Dataset"
y_label = r"TQ of Dataset"
env = "MountainCar-v0"
userun = 1

online_return = np.max(data[env][userun]["online"]["DQN"])
random_return = mm.get_data(env, "random", userun)[0][0]
online_usap = mm.get_data(env, "er", userun)[2]

x, y = [], []
for m, mode in enumerate(["random", "fully", "mixed", "er", "noisy"]):
    x.append(mm.get_data(env, mode, userun)[2] / online_usap)
    y.append((mm.get_data(env, mode, userun)[0][0] - random_return) / (online_return - random_return))

f = plt.figure(figsize=(6, 5))

plt.scatter(x, y, s = 70)
plt.xlim(-0.02, 1.12)
plt.ylim(-0.02, 1.02)

for i, annotation in enumerate(['Random', 'Expert', 'Mixed', 'Replay  ', 'Noisy']):
    plt.annotate(annotation,
                 (x[i], y[i] + offset_ann - 0.002),
                 fontsize="medium", va="bottom", ha="center",zorder=20)


f.tight_layout(rect=(0.022, 0.022, 1, 1))
f.text(0.5, 0.01, x_label, ha='center', fontsize="large")
f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
plt.savefig(os.path.join(outdir, f"mountaincar_tqvssaco." + image_type))
plt.close()

### Other envs
outdir = os.path.join("..", "..", "results", folder, "projections")
os.makedirs(outdir, exist_ok=True)

samples = 10000

np.random.seed(42)

for e, env in enumerate(list(envs.keys())[:4]):
    ind = np.random.choice(2 * 10 ** 6 if "MinAtar" in env else 10 ** 5, (samples,), replace=False)
    f, axs = plt.subplots(5, 5, figsize=figsize_theplot, sharex=True, sharey=True)
    axs = [item for sublist in zip(axs[0], axs[1], axs[2], axs[3], axs[4]) for item in sublist]

    with open(os.path.join("..", "..", "data", f"ex{e + 1}", f"{env}_run1_random.pkl"), "rb") as file:
        ds = pickle.load(file)

    proj = np.random.randn(len(ds.state[0]), 2)

    for m, bt in enumerate(buffer):
        for userun in range(1, useruns + 1):
            ax = axs[m * 5 + userun - 1]
            # load saved buffer
            with open(os.path.join("..", "..", "data", f"ex{e + 1}", f"{env}_run{userun}_{bt}.pkl"), "rb") as file:
                ds = pickle.load(file)

                if userun == 1:
                    ax.set_title(buffer[bt])
                if e == 1:
                    ax.scatter(ds.state[ind, 0], ds.state[ind, 1], c=[f"C{a}" for a in ds.action[ind, 0]], s=0.5, alpha=0.5)
                else:
                    state = ds.state[ind] @ proj
                    ax.scatter(state[:, 0], state[:, 1], c=[f"C{a}" for a in ds.action[ind, 0]], s=1, alpha=0.25)
                ax.text(0.02, 0.92, f"Run {userun}", fontsize="small", transform=ax.transAxes)

    f.tight_layout(rect=(0.022, 0.022, 1, 0.98))
    f.text(0.53, 0.98, "Dataset", ha='center', fontsize="large")
    f.text(0.53, 0.01, "Axis 1", ha='center', fontsize="large")
    f.text(0.005, 0.5, "Axis 2", va='center', rotation='vertical', fontsize="large")
    plt.savefig(os.path.join(outdir, f"{env}.png"))
    plt.close()
