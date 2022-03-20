import os
import glob
import shutil
import pickle
import numpy as np
from source.offline_ds_evaluation.metrics_manager import MetricsManager
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
# Turn interactive plotting off
plt.ioff()
import seaborn as sns
sns.set()

matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)

image_type = "pdf"
figsize = (5.5, 4)

mm = MetricsManager(0)
useruns = 5

experiments = ["ex101", "ex102", "ex103", "ex104", "ex105", "ex106"]


envs = ['Breakout-MinAtar-v0', 'RotateShifted-Breakout-MinAtar-v0', 'Homomorph-Breakout-MinAtar-v0',
        'DistShift-Breakout-MinAtar-v0', 'SpaceInvaders-MinAtar-v0', 'MiniGrid-Dynamic-Obstacles-8x8-v0']

############# Create files out of run log files

runs = 1

algos = ["BC", "BVE", "MCE", "DQN", "QRDQN", "REM", "BCQ", "CQL", "CRR"]
algo = "DQN"
datasets = ["random", "mixed", "er", "noisy", "fully"]

origin = os.path.join("..", "..", "results", "raw")
target = os.path.join("..", "..", "results", "csv_per_userun")
os.makedirs(target, exist_ok=True)

folders = ["avd", "return"]

os.makedirs(target, exist_ok=True)
for e, env in enumerate(envs):
    for folder in folders:
        for userun in range(useruns):
            os.makedirs(os.path.join(target, folder, env, f"userun{userun + 1}"), exist_ok=True)
        for run in range(useruns):
            shutil.copy(os.path.join(origin, folder, experiments[e], f"{env}_online_run{run + 1}.csv"),
                        os.path.join(target, folder, env, f"userun{run + 1}", f"DQN_online.csv"))

#############


## get data
indir = os.path.join("..", "..", "results", "csv_per_userun", "return")

files = []
for file in glob.iglob(os.path.join(indir, "**", "*.csv"), recursive=True):
    for env in envs:
        if env in file:
            files.append(file)

data = dict()
for file in files:
    name = file.split("\\")[-1]
    userun = int(file.split("\\")[-2][-1])
    env = file.split("\\")[-3]
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




buffer = {"random": "Random", "mixed": "Mixed", "er": "Replay", "noisy": "Noisy", "fully": "Expert"}
modes = list(buffer.keys())



for e, ex in enumerate(experiments):
    for userun in range(1, 6):
        paths = glob.glob(os.path.join("..", "..", "data", ex, f"metrics_*_run{userun}.pkl"))
        for path in paths:
            with open(path, "rb") as f:
                m = pickle.load(f)
                if ex == "ex15":
                    for key in list(m.data.keys()):
                        try:
                            _, buffer_type = key.split("/")
                            m.data["/".join([envs[e], buffer_type])] = m.data.pop(key)
                        except KeyError:
                            pass
                m.recode(userun)
            mm.data.update(m.data)



comps = [[envs[0]], [envs[0], envs[1]], [envs[0], envs[2]], [envs[0], envs[3]], [envs[0], envs[4]], [envs[0], envs[5]]]
    

markers = ["o", "D", "*", "<", "s"]
normalize = Normalize(vmin=0, vmax=120, clip=True)

for c, comp in enumerate(comps):
    f, axs = plt.subplots(1, 1, figsize=figsize, sharex=True, sharey=True)
    #axs = [item for sublist in zip(axs[0], axs[1]) for item in sublist]


    names = [Patch(facecolor="C0"),
             Patch(facecolor="C1")]

    datasets = [Line2D([0], [0], color="black", marker=markers[i], linewidth=0) for i in range(len(markers))]

    for e, env in enumerate(comp):
        ax = axs

        ax.axhline(y=1, color="silver")
        ax.axvline(x=1, color="silver")

        for m, mode in enumerate(modes):
            x, y, performance = [], [], []
            for userun in range(1, useruns + 1):
                online_return = np.max(data[env][userun]["online"]["DQN"])
                random_return = mm.get_data(env, "random", userun)[0][0]
                online_usap = mm.get_data(env, "er", userun)[2]

                x.append(mm.get_data(env, mode, userun)[2] / online_usap)
                y.append((mm.get_data(env, mode, userun)[0][0] - random_return) / (online_return - random_return))

            ax.scatter(x, y, s=60, c=f"C{e}", marker=markers[m], zorder=10, alpha=0.25)
            ax.scatter(np.mean(x), np.mean(y), s=180, c=f"C{e}", marker=markers[m], zorder=10)

    outdir = os.path.join("..", "..", "results", "main_figures_paper", "ablations")
    os.makedirs(outdir, exist_ok=True)
    x_label = r"SACo of Dataset"
    y_label = r"TQ of Dataset"

    f.tight_layout(rect=(0.022, 0.022, 1, 1))
    ax.set_ylim(bottom=-0.05, top=1.05)
    ax.set_xlim(left=-0.05, right=1.05)
    legend1 = ax.legend(names, [env.replace("-v0", "") for env in comp], loc="upper right")
    #f.text(0.5, 0.96, f"Datasets for DistShift1 and DistShift2", ha='center', fontsize="x-large")
    f.text(0.5, 0.01, x_label, ha='center', fontsize="large")
    f.text(0.005, 0.5, y_label, va='center', rotation='vertical', fontsize="large")
    ax.legend(datasets, [buffer[m] for m in modes], loc="lower right")
    plt.gca().add_artist(legend1)
    plt.savefig(os.path.join(outdir, f"ex10{c+1}." + image_type))
    plt.close()



