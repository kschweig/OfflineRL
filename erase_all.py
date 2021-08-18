import os
import shutil

shutil.rmtree("data")
shutil.rmtree("models")
shutil.rmtree("results")
shutil.rmtree("runs")

# create necessary folders
os.makedirs("runs")
os.makedirs("data")
os.makedirs("models")
os.makedirs("results")