import os
import glob
import pandas
import pickle
import numpy as np
from source.offline_ds_evaluation.metrics_manager import MetricsManager
import plotly.express as px

########################################################################################################################
#####DATA ACQUISITION
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

########################################################################################################################

folder = "main_figures_paper"
image_type = "pdf"
outdir = os.path.join("..", "..", "results", folder, "parallel_coordinates")

x = np.zeros((2 + len(algos), 4 * len(buffer) * useruns))

for e, env in enumerate(list(envs.keys())[:4]):
    for a, algo in enumerate(algos):
        for b, mode in enumerate(buffer):
            for userun in range(1, useruns + 1):
                online_return = np.max(data[env][userun]["online"]["DQN"])
                random_return = mm.get_data(env, "random", userun)[0][0]
                online_usap = mm.get_data(env, "er", userun)[2]

                max_algo_return = np.max(np.mean(data[env][userun][mode][algo], axis=1))
                tq = (mm.get_data(env, mode, userun)[0][0] - random_return) / (online_return - random_return)
                saco = mm.get_data(env, mode, userun)[2] / online_usap

                x[0, e * len(buffer) * useruns + b * useruns + userun - 1] = tq
                x[1, e * len(buffer) * useruns + b * useruns + userun - 1] = saco
                x[2 + a, e * len(buffer) * useruns + b * useruns + userun - 1] = (max_algo_return - random_return) / (online_return - random_return)

labels = ["rTQ", "rSACo"]
labels.extend(algos)

df = pandas.DataFrame(data=x.transpose((1, 0)), columns=labels)
df.insert(0, "Environment", [env for _ in range(len(buffer) * useruns) for env in range(4)])
df.insert(1, "Dataset setting", [b for b in range(len(buffer))] * 4 * useruns)


fig = px.parallel_coordinates(data_frame=df, color="Environment",
                              color_continuous_scale=px.colors.sequential.Viridis)
fig.show()

fig = px.parallel_coordinates(data_frame=df, color="Dataset setting",
                              color_continuous_scale=px.colors.sequential.Viridis)
fig.show()

fig = px.parallel_coordinates(data_frame=df, color="rTQ",
                              color_continuous_scale=px.colors.sequential.Viridis)
fig.show()

fig = px.parallel_coordinates(data_frame=df, color="rSACo",
                              color_continuous_scale=px.colors.sequential.Viridis)
fig.show()