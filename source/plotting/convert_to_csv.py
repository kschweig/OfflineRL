import tensorflow as tf
import glob
import os
import re

##########################
#       Settings         #
##########################

# Which experiment to extract
import tensorflow.python.framework.errors_impl

ex = "ex4"
# Which tag should be extracted
#'eval/Reward (SMA)' 'eval/Entropy'
tags = ['eval/Reward (SMA)', 'eval/Action-Value deviation (mean) (SMA)']
# "reward" "entropy"
marks = ["return", "avd"]


for t, tag in enumerate(tags):
    os.chdir(os.path.join("..", "..", "runs", ex))
    files=[]
    outpath = os.path.join("..", "..", "results", "csv", marks[t])
    os.makedirs(outpath, exist_ok=True)


    for file in glob.glob(os.path.join("**", "*.tfevents.*"), recursive=True):
        files.append(file)
    files.sort()


    data = []
    for file in files:

        run = int(re.findall("[0-9]+", file.split("/")[3])[0])

        if run == 1 and data != []:
            with open(os.path.join(outpath, f"{env}_{mode}_{algo}.csv"), "w") as w:
                minlen = len(data[0])
                last_full = 0
                for i, line in enumerate(data):
                    if len(line) < minlen:
                        # seldom, there occurs the phenomenon that the last reward in the tffiles cannot be read.
                        # Then replace all with the values read before. Only a minor difference on 2k iterations.
                        print("Datapoint at iteration", i, "replaced.")
                        line = data[last_full - 1]
                    else:
                        last_full = i
                    w.write(";".join([str(l) for l in line]) + "\n")
            data = []

        env = file.split("/")[0]
        mode = file.split("/")[1]
        algo = file.split("/")[2]

        try:
            i = 0
            for e in tf.compat.v1.train.summary_iterator(file):
                for v in e.summary.value:
                    iteration = 0
                    if v.tag == tag:
                        if len(data) <= i:
                            data.append([v.simple_value])
                        else:
                            data[i].append(v.simple_value)
                        i += 1
        except:
            print(f"Error in obtaining summary from {env}/{mode}/{algo}/{run}, may not contain complete data")

    # write data collected in the last run
    with open(os.path.join(outpath, f"{env}_{mode}_{algo}.csv"), "w") as w:
        minlen = len(data[0])
        for i, line in enumerate(data):
            if len(line) < minlen:
                # seldom, there occurs the phenomenon that the last reward in the tffiles cannot be read.
                # Then replace all with the values read before. Only a minor difference on 2k iterations.
                print("Datapoint at iteration", i, "replaced.")
                line = data[i-1]
            w.write(";".join([str(l) for l in line])+"\n")
