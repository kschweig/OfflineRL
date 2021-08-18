import os
import glob
import scipy
import pickle
import numpy as np
from source.offline_ds_evaluation.metrics_manager import MetricsManager
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
# Turn interactive plotting off
plt.ioff()
import seaborn as sns
sns.set()

matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)

run = 4

folder = ["baselines", "offpolicy", "offline", "all", "presentation"][run]
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
experiments = ["ex4", "ex5", "ex6"]

mm = MetricsManager(0)


for ex in experiments:
    paths = glob.glob(os.path.join("..", "..", "data", ex, "metrics*.pkl"))
    for path in paths:
        with open(path, "rb") as f:
            m = pickle.load(f)
        mm.data.update(m.data)

# static stuff

envs = {'CartPole-v1': 0, 'MountainCar-v0': 1, "MiniGrid-LavaGapS7-v0": 2, "MiniGrid-Dynamic-Obstacles-8x8-v0": 3,
        'Breakout-MinAtar-v0': 4, "Space_invaders-MinAtar-v0": 5}

algolist = [["BC", "BVE", "MCE"],
            ["DQN", "QRDQN", "REM"],
            ["BCQ", "CQL", "CRR"],
            ["BC", "BVE", "MCE", "DQN", "QRDQN", "REM", "BCQ", "CQL", "CRR"],
            ["BC", "BVE", "DQN", "BCQ"]]
algos = algolist[run]

buffer = {"random": "Random Policy", "mixed": "Mixed Policy", "er": "Experience Replay",
          "noisy": "Noisy Policy", "fully": "Final Policy"}

y_bounds = {'CartPole-v1': (-15, 15), "MiniGrid-LavaGapS7-v0":(-0.5, 1.3), 'MountainCar-v0': (-50, 100),
            "MiniGrid-Dynamic-Obstacles-8x8-v0":(-1, 1), 'Breakout-MinAtar-v0': (-5, 25), "Space_invaders-MinAtar-v0": (-5, 25)}

metrics = {(0,0):"Return (dataset)", (0,1):"Return (std)",
           1:"Unique States", 2:"Unique State-Action Pairs",
           (3,0):"Entropy", (3,1):"Entropy (std)",
           (4,0):"Sparsity", (4,1): "Sparsity (std)",
           (5,0):"Episode Length", (5,1):"Episode Length (std)",
           }

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


####################################
#       Usual Return plots         #
####################################

mark = "return"

# titles
y_label = "Moving Average Return"
x_label = "Update Steps"

indir = os.path.join("..", "..", "results", "csv", mark)
outdir = os.path.join("..", "..", "results", folder, mark)
os.makedirs(outdir, exist_ok=True)

files = []
for file in glob.glob(os.path.join(indir, "*.csv")):
    files.append(file)

data = dict()

for file in files:
    name = file.split("/")[-1]
    env = "_".join(name.split("_")[:-2])
    mode = name.split("_")[-2]
    algo = name.split("_")[-1].split(".")[0]

    try:
        csv = np.loadtxt(file, delimiter=";")
    except:
        print("Error in ", env, mode, algo)

    if len(csv.shape) == 1:
        csv = csv.reshape(-1, 1)

    if not data.keys() or env not in data.keys():
        data[env] = dict()
    if not data[env].keys() or mode not in data[env].keys():
        data[env][mode] = dict()

    data[env][mode][algo] = csv

for e, env in enumerate(data.keys()):

    f, axs = plt.subplots(1, 5, figsize=figsize_small, sharex=True, sharey=True)
    #axs = [item for sublist in axs for item in sublist]

    for m, mode in enumerate(data[env].keys()):

        if mode == "online":
            continue

        ids = list(buffer.keys())
        ax = axs[ids.index(mode)]

        norm = mm.get_data(env, mode)[0][0]
        ax.axhline(y=norm, color="black", linestyle="dotted",
                   linewidth=2, label=("Behav." if m==0 else None))
                   
        csv = data[env]["online"]["DQN"]
        ax.axhline(y=csv.max(), color="black", linewidth=2)
        plt_csv(ax, csv, "Online", mode, color="black", set_label=m==0)

        for a, algo in enumerate(algos):
            csv = data[env][mode][algo]
            plt_csv(ax, csv, algo, mode, color=f"C{(a + run * 3 if run < 3 else a)}", set_label=m==0)

    for ax in axs[m:]:
        f.delaxes(ax)

    f.text(0.52, 0.92, "-".join(env.split("-")[:-1]), ha='center', fontsize="x-large")
    #f.legend(loc="upper center", ncol=len(algos) + 2, fontsize="small")
    f.tight_layout(rect=(0.008, 0.022, 1, 0.92))
    f.text(0.52, 0.02, x_label, ha='center', fontsize="large")
    f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
    plt.savefig(os.path.join(outdir, env + "." + "png"))

    if e == 0:
        for ax in axs:
            ax.set_visible(False)
        for text in f.texts:
            text.set_visible(False)
        f.set_size_inches(figsize_small[0] - 4, 0.4, forward=True)
        f.legend(loc="center", ncol=len(algos) + 2, fontsize="small")
        f.tight_layout()
        plt.savefig(os.path.join(outdir, "legend." + image_type))
    plt.close()

###############
# plot metrics for policies
###############

modes = list(buffer.keys())

outdir = os.path.join("..", "..", "results", folder, "metrics")
os.makedirs(outdir, exist_ok=True)

# titles
x_label = "Dataset"

# plot for discussion

f, axs = plt.subplots(2, 3, figsize=figsize, sharex=True)
axs = [item for sublist in zip(axs[:, 0], axs[:, 1], axs[:,2]) for item in sublist]

for m, metric in enumerate([(0, 0), 2, (3, 0), 1, (5, 0), (4, 0)]):

    for env in envs:
        x = []
        random_return = mm.get_data(env, "random")[0][0]
        for mode in modes:
            if m == 1 or m == 3:
                x.append(mm.get_data(env, mode)[metric])
            else:
                x.append(mm.get_data(env, mode)[metric[0]][metric[1]])

        if m == 0:
            csv = data[env]["online"]["DQN"]
            x = [ (x_ - random_return) / (np.max(csv) - random_return) for x_ in x]
            axs[m].axhline(y=1, color="silver")

        axs[m].plot(range(len(x)), x, "-o", label = "-".join(env.split("-")[:-1]) if m == 0 else None, zorder=20)

    if m == 1 or m == 3 or m == 4:
        axs[m].set_yscale('log')

    if m == 5:
        axs[m].set_ylim(0.74, 1.01)

    if m == 0:
        axs[m].set_ylabel("Normalized Return")
    else:
        axs[m].set_ylabel(metrics[metric])
    axs[m].set_xticks(range(len(modes)))
    axs[m].set_xticklabels([buffer[m] for m in modes], fontsize="x-small", rotation=15, rotation_mode="anchor")

f.legend(loc="upper center", ncol=len(env), fontsize="small")
f.tight_layout(rect=(0, 0.022, 1, 0.95))
f.text(0.52, 0.01, x_label, ha='center', fontsize="large")
plt.savefig(os.path.join(outdir, "overview_6." + image_type))
plt.close()

# plot for thesis

f, axs = plt.subplots(1, 3, figsize=figsize_half, sharex=True)

for m, metric in enumerate([(0, 0), 2, (3, 0)]):

    for env in envs:
        x = []
        random_return = mm.get_data(env, "random")[0][0]
        online_usap = mm.get_data(env, "er")[2]
        for mode in modes:
            if m == 1 or m == 3:
                x.append(mm.get_data(env, mode)[metric])
            else:
                x.append(mm.get_data(env, mode)[metric[0]][metric[1]])

        if m == 0:
            csv = data[env]["online"]["DQN"]
            x = [(x_ - random_return) / (np.max(csv) - random_return) for x_ in x]
            axs[m].axhline(y=1, color="silver")
        if m == 1:
            x = [x_ / online_usap for x_ in x]
            axs[m].axhline(y=1, color="silver")

        axs[m].plot(range(len(x)), x, "-o", label = "-".join(env.split("-")[:-1]) if m == 0 else None, zorder=20)

    if m == 0:
        axs[m].set_ylabel("Relative Trajectory Quality")
    elif m == 1:
        axs[m].set_ylabel("Relative State-Action Coverage")
    else:
        axs[m].set_ylabel(metrics[metric])
    axs[m].set_xticks(range(len(modes)))
    axs[m].set_xticklabels([buffer[m] for m in modes], fontsize="x-small", rotation=15, rotation_mode="anchor")

f.legend(loc="upper center", ncol=len(env), fontsize="small")
f.tight_layout(rect=(0, 0.022, 1, 0.92))
f.text(0.52, 0.01, x_label, ha='center', fontsize="large")
plt.savefig(os.path.join(outdir, "overview_3." + image_type))
plt.close()

# plot for presentation

f, axs = plt.subplots(1, 2, figsize=figsize_half_half, sharex=True)

for m, metric in enumerate([(0, 0), 2]):

    for env in envs:
        x = []
        random_return = mm.get_data(env, "random")[0][0]
        online_usap = mm.get_data(env, "er")[2]
        for mode in modes:
            if m == 1 or m == 3:
                x.append(mm.get_data(env, mode)[metric])
            else:
                x.append(mm.get_data(env, mode)[metric[0]][metric[1]])

        if m == 0:
            csv = data[env]["online"]["DQN"]
            x = [(x_ - random_return) / (np.max(csv) - random_return) for x_ in x]
            axs[m].axhline(y=1, color="silver")
        if m == 1:
            x = [x_ / online_usap for x_ in x]
            axs[m].axhline(y=1, color="silver")

        axs[m].plot(range(len(x)), x, "-o", label = "-".join(env.split("-")[:-1]) if m == 0 else None, zorder=20)

    if m == 0:
        axs[m].set_ylabel("Relative Trajectory Quality")
    elif m == 1:
        axs[m].set_ylabel("Relative State-Action Coverage")
    axs[m].set_xticks(range(len(modes)))
    axs[m].set_xticklabels([buffer[m] for m in modes], fontsize="x-small", rotation=15, rotation_mode="anchor")

f.legend(loc="upper center", ncol=(len(envs) // 2), fontsize="x-small")
f.tight_layout(rect=(0, 0.022, 1, 0.88))
f.text(0.52, 0.01, x_label, ha='center', fontsize="large")
plt.savefig(os.path.join(outdir, "overview_2." + image_type))
plt.close()

#############################
#        Comparisons        #
#############################

##################
# load reward data
##################
indir = os.path.join("..", "..", "results", "csv", "return")
outdir = os.path.join("..", "..", "results", folder, "comp_return")
os.makedirs(outdir, exist_ok=True)

files = []
for file in glob.glob(os.path.join(indir, "*.csv")):
    files.append(file)

data = dict()

for file in files:
    name = file.split("/")[-1]
    env = "_".join(name.split("_")[:-2])
    mode = name.split("_")[-2]
    algo = name.split("_")[-1].split(".")[0]

    try:
        csv = np.loadtxt(file, delimiter=";")
    except:
        print("Error in ", env, mode, algo)

    if len(csv.shape) == 1:
        csv = csv.reshape(-1, 1)

    # first hundred invalid, as they are not the correct sma!
    csv = csv[100:]

    if not data.keys() or env not in data.keys():
        data[env] = dict()
    if not data[env].keys() or mode not in data[env].keys():
        data[env][mode] = dict()

    data[env][mode][algo] = (np.mean(csv, axis=1).max(), np.std(csv, axis=1)[np.argmax(np.mean(csv, axis=1))])

###############
# plot metrics + policy for reward
###############

# titles
y_label = "Maximum Moving Average Return"
x_label = "Dataset"


for metric in metrics.keys():

    f, axs = plt.subplots(2, 3, figsize=figsize_comp, sharex=(metrics[metric] == "Entropy"))
    axs = [item for sublist in zip(axs[0], axs[1]) for item in sublist]

    for e, env in enumerate(envs):

        ax = axs[e]
        ax.set_title(env[:-3])

        for a, algo in enumerate(algos):
            x, y, sd = [], [], []
            for mode in modes:
                if metric == 1 or metric == 2:
                    x.append(mm.get_data(env, mode)[metric])
                else:
                    x.append(mm.get_data(env, mode)[metric[0]][metric[1]])
                y.append(data[env][mode][algo][0])
                sd.append(data[env][mode][algo][1])

            x, y, sd = [list(tuple) for tuple in zip(*sorted(zip(x, y, sd)))]

            cis = (np.asarray(y) - np.asarray(sd), np.asarray(y) + np.asarray(sd))
            ax.fill_between(x, cis[0], cis[1], alpha=0.2, color=f"C{(a + run * 3 if run < 3 else a)}")
            ax.plot(x, y, "o-", label=(algo if e == 1 else None), color=f"C{(a + run * 3 if run < 3 else a)}")

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

# plot for modes
f, axs = plt.subplots(2, 3, figsize=figsize_comp, sharex=True)
axs = [item for sublist in zip(axs[0], axs[1]) for item in sublist]

for e, env in enumerate(envs):

    ax = axs[e]
    ax.set_title(env[:-3])

    x, y = list(range(len(buffer))), []
    for mode in modes:
        y.append(mm.get_data(env, mode)[0][0])
    x, y = [list(tuple) for tuple in zip(*sorted(zip(x, y)))]

    ax.plot(x, y, "o-", linestyle="dotted", label=("Behav." if e == 0 else None), color="black")

    # Online Policy
    csv = data[env]["online"]["DQN"]
    ax.axhline(y=csv[0], color="black", label=("Online" if e == 0 else None))

    for a, algo in enumerate(algos):

        x, y, sd = [], [], []
        for m, mode in enumerate(modes):
            x.append(m)
            y.append(data[env][mode][algo][0])
            sd.append(data[env][mode][algo][1])
        x, y, sd = [list(tuple) for tuple in zip(*sorted(zip(x, y, sd)))]

        cis = (np.asarray(y) - np.asarray(sd), np.asarray(y) + np.asarray(sd))
        ax.fill_between(x, cis[0], cis[1], alpha=0.2, color=f"C{(a + run * 3 if run < 3 else a)}")
        ax.plot(x, y, "o-", label=(algo if e == 0 else None), color=f"C{(a + run * 3 if run < 3 else a)}")

    x = []
    for m, mode in enumerate(modes):
        x.append(m)

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([buffer[m] for m in modes], fontsize="x-small", rotation=15, rotation_mode="anchor")

f.legend(loc="upper center", ncol=len(algos) + 2, fontsize="small")
f.tight_layout(rect=(0.008, 0.022, 1, 0.95))
f.text(0.52, 0.01, x_label, ha='center', fontsize="large")
f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
plt.savefig(os.path.join(outdir, "buffertypes." + image_type))
plt.close()

# comparison plot
f, axs = plt.subplots(2, 3, figsize=figsize_comp, sharex=True)
axs = [item for sublist in zip(axs[0], axs[1]) for item in sublist]

for e, env in enumerate(envs):

    ax = axs[e]
    ax.set_title(env[:-3])

    x, y = list(range(len(buffer))), []
    for mode in modes:
        y.append(mm.get_data(env, mode)[0][0])
    x, y = [list(tuple) for tuple in zip(*sorted(zip(x, y)))]

    ax.plot(x, y, "o-", linestyle="dotted", label=("Behav." if e == 0 else None), color="black")

    # Online Policy
    csv = data[env]["online"]["DQN"]
    ax.axhline(y=csv[0], color="black", label=("Online" if e == 0 else None))

    x, y = [], []

    for a, algo in enumerate(algos):

        x_, y_ = [], []
        for m, mode in enumerate(modes):
            x_.append(m)
            y_.append(data[env][mode][algo][0])
        x_, y_ = [list(tuple) for tuple in zip(*sorted(zip(x_, y_)))]
        x.append(x_)
        y.append(y_)

    x = np.asarray(x).mean(axis=0)
    sd = np.asarray(y).std(axis=0)
    y = np.asarray(y).mean(axis=0)

    cis = (y - sd, y + sd)
    ax.fill_between(x, cis[0], cis[1], alpha=0.2, color=f"C9")
    ax.plot(x, y, "o-", label=("Average Performance" if e == 0 else None), color=f"C0")

    x = []
    for m, mode in enumerate(modes):
        x.append(m)

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([buffer[m] for m in modes], fontsize="x-small", rotation=15, rotation_mode="anchor")

f.legend(loc="upper center", ncol=len(algos) + 2, fontsize="small")
f.tight_layout(rect=(0.008, 0.022, 1, 0.95))
f.text(0.52, 0.01, x_label, ha='center', fontsize="large")
f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
plt.savefig(os.path.join(outdir, "compare_buffertypes." + image_type))
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

sizes = [(1, 3), (1, 3), (1, 3), (3, 3), (2, 2)]

figsizes = [(figsize_theplot[0] / 3, figsize_theplot[1]),
            (figsize_theplot[0] / 3, figsize_theplot[1]),
            (figsize_theplot[0] / 3, figsize_theplot[1]),
            (figsize_theplot[0], figsize_theplot[1]),
            (figsize_theplot[0] / 3 * 2, figsize_theplot[1] / 3 * 2)]

### for algos

f, axs = plt.subplots(sizes[run][0], sizes[run][1], figsize=figsizes[run], sharex=True, sharey=True)
if run == 4:
    axs = [item for sublist in zip(axs[:, 0], axs[:, 1]) for item in sublist]
elif run == 3:
    axs = [item for sublist in zip(axs[:, 0], axs[:, 1], axs[:, 2]) for item in sublist]
else:
    pass

for a, algo in enumerate(algos):
    ax = axs[a]

    ax.axhline(y=1, color="silver")
    ax.axvline(x=1, color="silver")

    x, y, performance = [], [], []

    for env in envs:
        ax.set_title(algo, fontsize="large")

        online_return = data[env]["online"]["DQN"][0]
        random_return = mm.get_data(env, "random")[0][0]
        online_usap = mm.get_data(env, "er")[2]

        for m, mode in enumerate(modes):
            x.append(mm.get_data(env, mode)[2] / online_usap)
            y.append( (mm.get_data(env, mode)[0][0] - random_return) / (online_return - random_return))
            performance.append((data[env][mode][algo][0] - random_return) / (online_return - random_return) * 100)

    ax.scatter(x, y, s = 100, c=performance, cmap=cmap, norm=normalize, zorder=10)

    for i in range(len(performance)):
        ax.annotate(f"{int(performance[i])}%", (x[i] + offset_ann, y[i] + offset_ann), fontsize="x-small", zorder=20)

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

for method in ["Mean", "Maximum"]:

    f, axs = plt.subplots(2, 3, figsize=figsize_envs, sharex=True, sharey=True)
    axs = [item for sublist in zip(axs[0], axs[1]) for item in sublist]

    for e, env in enumerate(envs):
        ax = axs[e]

        ax.axhline(y=1, color="silver")
        ax.axvline(x=1, color="silver")

        online_return = data[env]["online"]["DQN"][0]
        random_return = mm.get_data(env, "random")[0][0]
        online_usap = mm.get_data(env, "er")[2]

        x, y, performance = [], [], []
        for m, mode in enumerate(modes):
            x.append(mm.get_data(env, mode)[2] / online_usap)
            y.append((mm.get_data(env, mode)[0][0] - random_return) / (online_return - random_return))

        for algo in algos:
            ax.set_title("-".join(env.split("-")[:-1]), fontsize="large")

            p = []
            for m, mode in enumerate(modes):
                p.append((data[env][mode][algo][0] - random_return) / (online_return - random_return) * 100)
            performance.append(p)

        if method == "Mean":
            performance = np.mean(np.asarray(performance), axis=0)
        elif method == "Maximum":
            performance = np.max(np.asarray(performance), axis=0)

        ax.scatter(x, y, s = 100, c=performance, cmap=cmap, norm=normalize, zorder=10)

        for i in range(len(performance)):
            ax.annotate(f"{int(performance[i])}%", (x[i] + offset_ann, y[i] + offset_ann), fontsize="x-small", va="bottom", ha="left",zorder=20)
            ax.annotate(annotations[i], (x[i] - offset_ann, y[i] + offset_ann), fontsize="x-small", va="bottom", ha="right", zorder=30)

    f.tight_layout(rect=(0.022, 0.022, 1, 0.96))
    f.text(0.53, 0.96, f"{method} performance across algorithms", ha='center', fontsize="x-large")
    f.text(0.53, 0.01, x_label, ha='center', fontsize="large")
    f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
    plt.savefig(os.path.join(outdir, f"envs_{method}." + image_type))
    plt.close()

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
