# Understanding the Effects of Datasets Characteristics on Offline Reinforcement Learning
_Kajetan Schweighofer<sup>1</sup>,
Markus Hofmarcher<sup>1</sup>,
Markus-Constantin Dinu<sup>1,3</sup>,
Philipp Renz<sup>1</sup>,
Angela Bitto-Nemling<sup>1</sup>,
Vihang Patil<sup>1</sup>,
Sepp Hochreiter<sup>1, 2</sup>_

<sup>1</sup> ELLIS Unit Linz and LIT AI Lab, Institute for Machine Learning, Johannes Kepler University Linz, Austria  
<sup>2</sup> Institute of Advanced Research in Artificial Intelligence (IARAI)  
<sup>3</sup> Dynatrace Research, Austria

---

The paper is available on [arxiv](https://arxiv.org/abs/2111.04714)

---

# Implementation
This repository contains implementations of BC, BVE, MCE, DQN, QR-DQN, REM, BCQ, CQL and CRR,
used for our evaluation of Offline RL datasets.
Implementation-wise, algorithms can in theory be used in the usual Online RL setting as well as Offline RL settings.
Furthermore, utilities for offline dataset evaluation and plotting of results are contained.

Experiments are managed through experimental files (ex_01.py, ex_02.py, ...).
While this is not a necessity, we created an experimental file for each of the six environments
used to obtain our results, to more easily distribute experiments across multiple devices.

## Dependencies
To reproduce all results we provide an environment.yml file to setup a conda environment with the required packages.
Run the following command to create and activate the environment:

```shell script
conda env create --file environment.yml
conda activate offline_rl
pip install -e .
```

## Usage

To create datasets for Offline RL, each experimental file needs to be run by

```shell script
python ex_XX.py --online
```

After this run has finished, datasets for Offline RL are created, which are then used for applying algorithms in the Offline RL setting.
Offline experiments are started with

```shell script
python ex_XX.py
```

Runtimes will be long, especially on MinAtar environments, which is why distribution across multiple machines is crucial in this step.
To distribute across multiple machines, two further command line arguments are eligible, ```--run``` and ```--dataset```.
Depending on how many runs have been done to create datasets for Offline RL (five in the paper), one can select a specific version of the dataset
with the first parameter.
For the results in the paper, five different datasets are created (random, mixed, replay, noisy, expert), which can be selected
by its number using the second parameter.

As an example, offline experiments using the fourth dataset creation run on the expert dataset is started with

```shell script
python ex_XX.py --run 3 --dataset 4
```

or using the first dataset creation run on the replay dataset

```shell script
python ex_XX.py --run 0 --dataset 2
```
## Results
After all experiments are concluded, one has to combine the logged files and create the plots by executing

```shell script
python source/plotting/join_csv_files.py
python source/plotting/create_plots.py
```

Furthermore, plots for the training curves can be created by executing

```shell script
python source/plotting/learning_curves.py
```

Alternative visualisations of the main results, using parallel coordinates are available by executing

```shell script
python source/plotting/parallel_coordinates.py
```

## LICENSE
MIT LICENSE
