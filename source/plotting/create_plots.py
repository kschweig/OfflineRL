import os
import glob
import scipy
import pickle
import numpy as np
from source.offline_ds_evaluation.metrics_manager import MetricsManager
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
# Turn interactive plotting off
plt.ioff()
import seaborn as sns
sns.set()

matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)

folder="main_figures"
image_type = "png"

figsize = (12, 6)
figsize_legend = (12, 1)
figsize_half = (12, 3.5)
figsize_half_half = (8, 4)
figsize_small = (16, 3)
figsize_comp = (12, 6)
figsize_envs = (12, 7.2)
figsize_theplot = (12, 12)

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

buffer = {"random": "Random Policy", "mixed": "Mixed Policy", "er": "Experience Replay",
          "noisy": "Noisy Policy", "fully": "Final Policy"}

metrics = {(0,0):"Return (dataset)", 2:"Unique State-Action Pairs", (3,0):"Entropy"}

annotations = ["(R)", "(M)", "(E)", "(N)", "(F)"]


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

modes = list(buffer.keys())

outdir = os.path.join("..", "..", "results", folder, "metrics")
os.makedirs(outdir, exist_ok=True)

# titles
x_label = "Dataset"

# dataset overview

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

        x = np.asarray(x_all)
        est = np.mean(x, axis=1)
        sd = np.std(x, axis=1)
        cis = (est - sd, est + sd)

        axs[m].fill_between(range(len(x)), cis[0], cis[1], alpha=0.2)

        axs[m].plot(range(len(est)), est, "o",
                    label = "-".join(env.split("-")[:-1]) if m == 0 else None, zorder=20)

    if m == 0:
        axs[m].set_ylabel("Relative Trajectory Quality")
        axs[m].axhline(y=1, color="silver")
    elif m == 1:
        axs[m].set_ylabel("Relative State-Action Coverage")
        axs[m].axhline(y=1, color="silver")
    else:
        axs[m].set_ylabel(metrics[metric])

    axs[m].set_ylim(bottom=-0.05, top=1.45)
    axs[m].set_xticks(range(len(modes)))
    axs[m].set_xticklabels([buffer[m] for m in modes], fontsize="x-small", rotation=15, rotation_mode="anchor")
axs[-1].set_ylim(bottom=-0.05, top=1.05)
f.legend(loc="upper center", ncol=len(env), fontsize="small")
f.tight_layout(rect=(0, 0.022, 1, 0.92))
f.text(0.52, 0.01, x_label, ha='center')
plt.savefig(os.path.join(outdir, f"overview_3_line." + image_type))
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
        axs[m].set_ylabel("Relative Trajectory Quality")
        axs[m].axhline(y=1, color="silver")
    elif m == 1:
        axs[m].set_ylabel("Relative State-Action Coverage")
        axs[m].axhline(y=1, color="silver")
    else:
        axs[m].set_ylabel(metrics[metric])

    axs[m].set_ylim(bottom=-0.05, top=1.45)
    axs[m].set_xticks([i for i in range(len(modes) + 1)])
    names = [buffer[m] for m in modes]
    names.append("")
    axs[m].set_xticklabels(names,fontsize="x-small", rotation=15, rotation_mode="anchor")
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
        axs[m].set_ylabel("Relative Trajectory Quality")
        axs[m].axhline(y=1, color="silver")
    elif m == 1:
        axs[m].set_ylabel("Relative State-Action Coverage")
        axs[m].axhline(y=1, color="silver")

    axs[m].set_ylim(bottom=-0.05, top=1.45)
    axs[m].set_xticks([i for i in range(len(modes) + 1)])
    names = [buffer[m] for m in modes]
    names.append("")
    axs[m].set_xticklabels(names, fontsize="x-small", rotation=15, rotation_mode="anchor")

labels = [mpatches.Patch(color=f"C{e_}", fill=False, linewidth=1.5, label="-".join(env_.split("-")[:-1])) for e_, env_ in enumerate(envs)]
f.legend(handles=labels, handlelength=1, loc="upper center", ncol=len(envs)//2, fontsize="small")
f.tight_layout(rect=(0, 0.022, 1, 0.88))
f.text(0.52, 0.01, x_label, ha='center')
plt.savefig(os.path.join(outdir, f"overview_2." + image_type))
plt.close()


###############
# One plot to rule them all, one plot to find them, one plot to bring them all and in the darkness bind them
###############

outdir = os.path.join("..", "..", "results", folder, "tq_vs_sac")
os.makedirs(outdir, exist_ok=True)

from  matplotlib.colors import LinearSegmentedColormap
c = ["red", "tomato", "lightsalmon", "moccasin", "palegreen", "limegreen", "green"]
v = [0 ,.15 ,.4 ,.5,0.6,.9,1.]
l = list(zip(v, c))
cmap = LinearSegmentedColormap.from_list('rg',l, N=256)
normalize = Normalize(vmin=30, vmax=130, clip=True)

offset_ann = 0.025

# titles
x_label = r"Relative $\bf{{State}{-}{Action} \; Coverage}$ to Online Policy"
y_label = r"Relative $\bf{Trajectory \; Quality}$ to Online Policy"

# plot for discussion

### for algos

f, axs = plt.subplots(3, 3, figsize=(figsize_theplot[0], figsize_theplot[1]), sharex=True, sharey=True)
axs = [item for sublist in zip(axs[:, 0], axs[:, 1], axs[:, 2]) for item in sublist]


for a, algo in enumerate(algos):
    ax = axs[a]

    ax.axhline(y=1, color="silver")
    ax.axvline(x=1, color="silver")
    ax.set_title(algo, fontsize="large")

    x, y, performance = [], [], []

    for e, env in enumerate(envs):
        for userun in range(1, useruns + 1):

            online_return = np.max(data[env][userun]["online"]["DQN"])
            random_return = mm.get_data(env, "random", userun)[0][0]
            online_usap = mm.get_data(env, "er", userun)[2]

            for m, mode in enumerate(modes):
                try:
                    performance.append((np.max(np.mean(data[env][userun][mode][algo], axis=1)) - random_return) / (
                                online_return - random_return) * 100)
                    x.append(mm.get_data(env, mode, userun)[2] / online_usap)
                    y.append((mm.get_data(env, mode, userun)[0][0] - random_return) / (online_return - random_return))
                except:
                    continue



    ax.scatter(x, y, s = 100, c=performance, cmap=cmap, norm=normalize, zorder=10)

    """
    for i in range(len(performance)):
        ax.annotate(f"{int(performance[i])}%", (x[i] + offset_ann, y[i] + offset_ann), fontsize="x-small", zorder=20)
    """
    if a == 0:
        print("(TQ - SAC):", scipy.stats.pearsonr(x, y))
        print("-" * 30)
    print(algo, " (TQ - P):", scipy.stats.pearsonr(y, performance))
    print(algo, " (SAC - P):", scipy.stats.pearsonr(x, performance))
    print("-" * 30)


f.tight_layout(rect=(0.022, 0.022, 1, 1))
f.text(0.52, 0.01, x_label, ha='center', fontsize="large")
f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
plt.savefig(os.path.join(outdir, "algos." + image_type))
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

                    for m, mode in enumerate(modes):
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

    f.tight_layout(rect=(0.022, 0.022, 1, 0.96))
    f.text(0.53, 0.96, f"{method} performance across algorithms", ha='center', fontsize="x-large")
    f.text(0.53, 0.01, x_label, ha='center', fontsize="large")
    f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
    plt.savefig(os.path.join(outdir, f"envs_{method}." + image_type))
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
y_label = "Maximum Moving Average Return"
x_label = "Dataset"

"""
for metric in metrics.keys():

    f, axs = plt.subplots(2, 3, figsize=figsize_comp, sharex=(metrics[metric] == "Entropy"))
    axs = [item for sublist in zip(axs[0], axs[1]) for item in sublist]

    for e, env in enumerate(envs):

        ax = axs[e]
        ax.set_title(env[:-3])

        for a, algo in enumerate(algos):
            for userun in range(1, useruns + 1):
                x, y, sd = [], [], []
                for mode in modes:
                    if metric == 1 or metric == 2:
                        x.append(mm.get_data(env, mode, userun)[metric])
                    else:
                        x.append(mm.get_data(env, mode, userun)[metric[0]][metric[1]])
                    y.append(data[env][mode][algo])
                    sd.append(data[env][mode][algo][1])

                x, y, sd = [list(tuple) for tuple in zip(*sorted(zip(x, y, sd)))]

                cis = (np.asarray(y) - np.asarray(sd), np.asarray(y) + np.asarray(sd))
                ax.fill_between(x, cis[0], cis[1], alpha=0.2, color=f"C{a}")
                ax.plot(x, y, "o-", label=(algo if e == 1 else None), color=f"C{a}")

        x, y = [], []
        for mode in modes:
            if metric == 1 or metric == 2:
                x.append(mm.get_data(env, mode)[metric])
            else:
                x.append(mm.get_data(env, mode)[metric[0]][metric[1]])
            y.append(mm.get_data(env, mode)[0][0])
        x, y = [list(tuple) for tuple in zip(*sorted(zip(x, y)))]

        ax.plot(x, y, "o-", linestyle="dotted", label=("Behav." if e==0 else None), color="black")

        xmax, xmin, x_ = 0, 9e9, []
        for m, mode in enumerate(modes):
            if metric == 1 or metric == 2:
                x = mm.get_data(env, mode)[metric]
            else:
                x = mm.get_data(env, mode)[metric[0]][metric[1]]
            xmin = x if x < xmin else xmin
            xmax = x if x > xmax else xmax
            x_.append(x)

        # adjust markings if they overlap! do multiple times to be sure
        for _ in range(10):
            adjusted, no_changes = [], True
            for i in range(len(x_)):
                for j in range(len(x_)):
                    if i != j and i not in adjusted and abs(x_[i] - x_[j]) < 0.08 * (xmax - xmin):
                        delta = 0.08 * (xmax - xmin) - abs(x_[i] - x_[j])
                        if x_[i] < x_[j]:
                            x_[i] -= delta / 2
                            x_[j] += delta / 2
                        else:
                            x_[i] += delta / 2
                            x_[j] -= delta / 2
                        adjusted.append(j)
                        no_changes = False
            if no_changes:
                break

        # position text
        _, _, ymin, ymax = ax.axis()
        ax.set_ylim(ymin - (ymax - ymin) * 0.08, ymax)
        for m, x in enumerate(x_):
            ax.text(x, ymin - (ymax - ymin)*0.05, annotations[m], ha="center")

        # Online Policy
        csv = data[env]["online"]["DQN"]
        ax.axhline(y=csv[0], color="black", label=("Online" if e==0 else None))


    f.legend(loc="upper center", ncol=len(algos) + 2, fontsize="small")
    f.tight_layout(rect=(0.008, 0.022, 1, 0.95))
    f.text(0.52, 0.01, metrics[metric], ha='center', fontsize="large")
    f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
    plt.savefig(os.path.join(outdir, metrics[metric] + "." + image_type))
    plt.close()
"""

for userun in range(1, useruns + 1):
    # plot for modes
    f, axs = plt.subplots(2, 3, figsize=figsize_comp, sharex=True)
    axs = [item for sublist in zip(axs[0], axs[1]) for item in sublist]

    for e, env in enumerate(envs):

        ax = axs[e]
        ax.set_title(env[:-3])

        x, y = list(range(len(buffer))), []
        for mode in modes:
            y.append(mm.get_data(env, mode, userun)[0][0])
        x, y = [list(tuple) for tuple in zip(*sorted(zip(x, y)))]

        ax.plot(x, y, "o-", linestyle="dotted", label=("Behav." if e == 0 else None), color="black")

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
        ax.set_xticklabels([buffer[m] for m in modes], fontsize="x-small", rotation=15, rotation_mode="anchor")

    f.legend(loc="upper center", ncol=len(algos) + 2, fontsize="small")
    f.tight_layout(rect=(0.008, 0.022, 1, 0.95))
    f.text(0.52, 0.01, x_label, ha='center', fontsize="large")
    f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
    plt.savefig(os.path.join(outdir, f"buffertypes_userun{userun}." + image_type))
    plt.close()


"""
### for entropy

f = plt.figure(figsize=(6.2, 6))

plt.hlines(y=1, xmin=-0.05, xmax=1, color="silver")
plt.vlines(x=1, ymin=-0.05, ymax=1, color="silver")

plt.ylim(-0.05, 1.5)
plt.xlim(-0.05, 1.3)

tq, sac, en = [], [], []

for env in envs:

    online_return = data[env]["online"]["DQN"][0]
    random_return = mm.get_data(env, "random")[0][0]
    online_usap = mm.get_data(env, "er")[2]

    x, y, performance = [], [], []
    for m, mode in enumerate(modes):
        x.append(mm.get_data(env, mode)[2] / online_usap)
        y.append( (mm.get_data(env, mode)[0][0] - random_return) / (online_return - random_return))
        performance.append(mm.get_data(env, mode)[3][0])

    tq.extend(y)
    sac.extend(x)
    en.extend(performance)

    plt.scatter(x, y, s = 100, c=performance, cmap="Greens", zorder=10)

    for i in range(len(performance)):
        plt.annotate(f"{(performance[i]):.2f}", (x[i] + offset_ann, y[i] + offset_ann), fontsize="x-small", zorder=20)


f.tight_layout(rect=(0.04, 0.04, 1, 1))
f.text(0.54, 0.01, x_label, ha='center', fontsize="large")
f.text(0.01, 0.54, y_label, va='center', rotation='vertical', fontsize="large")
plt.savefig(os.path.join(outdir, "entropy." + image_type))
plt.close()

print("r (TQ - E):", scipy.stats.pearsonr(tq, en))
print("r (SAC - E):", scipy.stats.pearsonr(sac, en))
"""