from source.offline_ds_evaluation.metrics_manager import MetricsManager
import pickle
import glob
import os
import numpy as np


buffer = {"random": "Random", "mixed": "Mixed", "er": "Replay",
          "noisy": "Noisy", "fully": "Expert"}
modes = list(buffer.keys())

algos = ["BC", "BVE", "MCE", "DQN", "QRDQN", "REM", "BCQ", "CQL", "CRR"]

envs = {'CartPole-v1': 0, 'MountainCar-v0': 1, "MiniGrid-LavaGapS7-v0": 2, "MiniGrid-Dynamic-Obstacles-8x8-v0": 3,
        'Breakout-MinAtar-v0': 4, "SpaceInvaders-MinAtar-v0": 5}

experiments = ["ex1", "ex2", "ex3", "ex4", "ex5", "ex6"]
useruns = 5
runs = 5


# metric manager
mm = MetricsManager(0)


for ex in experiments:
    for userun in range(1, 6):
        paths = glob.glob(os.path.join("..", "..", "data", ex, f"metrics_*_run{userun}.pkl"))
        for path in paths:
            with open(path, "rb") as f:
                m = pickle.load(f)
                m.recode(userun)
            mm.data.update(m.data)

"""
### return per environment
for env in envs:
    for b in buffer:
        returns = []
        for userun in range(useruns):
            #returns.append(mm.get_data(env, b, userun+1)[0][0])
            returns.append(mm.get_data(env, b, userun + 1)[2])
        #print("\t".join([f"${ret:.2f}$" for ret in returns]))
        print("\t".join([f"${ret:,}$".replace(",", "\,") for ret in returns]))

for env in envs:
    returns = []
    for userun in range(useruns):
        returns.append(mm.get_data(env, "random", userun+1)[0][0])
    print("\t".join([f"${ret:.2f}$" for ret in returns]))
"""



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


"""
### maximum online policy
for env in envs:
    returns = []
    for userun in range(useruns):
        returns.append(np.max(data[env][userun + 1]["online"]["DQN"]))
    print(env, "\t".join([f"${ret:.2f}$" for ret in returns]))
    
### mean random policy
for env in envs:
    returns = []
    for userun in range(useruns):
        returns.append(np.max(data[env][userun + 1]["online"]["DQN"]))
    print(env, "\t".join([f"${ret:.2f}$" for ret in returns]))
"""

print("\\begin{table}[]\n \centering \n \\begin{tabular}{lccccccccc} \hline")
print("Dataset", "\t & \t", "\t & \t".join(algos), "\t \\\\")

for env in list(envs.keys())[:4]:
    print("\hline \multicolumn{10}{c}{" + env + "} \\\\ ")
    for m, mode in enumerate(modes):
        performance = []
        for algo in algos:

            p = []
            for userun in range(1, useruns + 1):
                online_return = np.max(data[env][userun]["online"]["DQN"])
                random_return = mm.get_data(env, "random", userun)[0][0]

                try:
                    p.append((np.max(np.mean(data[env][userun][mode][algo], axis=1)) - random_return) /
                                       (online_return - random_return))
                except:
                    break

            if p != []:
                performance.append((np.mean(p), np.std(p)))
        if performance != []:
            p_ = []
            for_std = np.argmax(np.asarray(performance)[:, 0])
            for mean, std in performance:
                if round(mean, 2) >= round(np.max(np.asarray(performance)[:, 0]), 2) - max(round(np.asarray(performance)[for_std, 1], 2), 0.05):
                    p_.append("\\begin{tabular}[c]{@{}c@{}}$\mathbf{" + f"{mean:.2f}" + "}" + f" $ \\\\ $\pm {std:.2f}$" + "\end{tabular}")
                else:
                    p_.append("\\begin{tabular}[c]{@{}c@{}}" + f"${mean:.2f} $ \\\\ $\pm {std:.2f}$" + "\end{tabular}")
            print(buffer[mode], "\t & \t", "\t & \t".join(p_), "\t \\\\")
print("\hline\n\end{tabular}\n\end{table}")

print("\n" * 3)

print("\\begin{table}[] \n \centering \n \\begin{tabular}{lcccc} \hline")
print("Dataset", "\t & \t", "\t & \t".join(["BC", "DQN", "BCQ", "CQL"]), "\t \\\\")

for env in list(envs.keys())[4:]:
    print("\hline \multicolumn{5}{c}{" + env + "} \\\\ ")
    for m, mode in enumerate(modes):
        performance = []
        for algo in algos:

            p = []
            for userun in range(1, useruns + 1):
                online_return = np.max(data[env][userun]["online"]["DQN"])
                random_return = mm.get_data(env, "random", userun)[0][0]

                try:
                    p.append((np.max(np.mean(data[env][userun][mode][algo], axis=1)) - random_return) /
                                       (online_return - random_return))
                except:
                    break
            if p != []:
                performance.append((np.mean(p), np.std(p)))
        if performance != []:
            p_ = []
            for_std = np.argmax(np.asarray(performance)[:, 0])
            for mean, std in performance:
                if round(mean, 2) >= round(np.max(np.asarray(performance)[:, 0]), 2) - max(round(np.asarray(performance)[for_std, 1], 2), 0.05):
                    p_.append("$\mathbf{" + f"{mean:.2f}" + "}" + f" \pm {std:.2f}$")
                else:
                    p_.append(f"${mean:.2f} \pm {std:.2f}$")
            print(buffer[mode], "\t & \t", "\t & \t".join(p_), "\t \\\\")
print("\hline\n\end{tabular}\n\end{table}")
