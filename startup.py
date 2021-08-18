import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
args = parser.parse_args()

# create necessary folders
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# launch tensorboard, can change launch folder if necessary
if args.e == 0:
    os.system('tensorboard --logdir=runs')
else:
    os.system(f'tensorboard --logdir=runs/ex{args.e}')