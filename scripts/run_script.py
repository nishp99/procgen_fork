import gym
import numpy as np
import math
import os
import submitit
import datetime
from utils import training_func
from utils import policy_network
from utils.policy_network import *
from utils.training_func import *
from procgen import ProcgenEnv
from utils import update
from utils.update import *
import sys
import os

# sys.path.append('utils')

#start timestamp with unique identifier for name
run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
#os.mkdir(with name of unique identifier)

#os.path.join(results, unique identifier)
results_path = os.path.join("utils", "results")
os.makedirs(results_path, exist_ok = True)

experiment_path = os.path.join(results_path, "n_or_more")
os.makedirs(experiment_path, exist_ok = True)

run_path = os.path.join(experiment_path, run_timestamp)
os.mkdir(run_path)

T = 4
n = 3
GAMMA = 0.9
episodes = 20*T
max_steps = 60
lr = 3e-4

executor = submitit.AutoExecutor(folder="utils/results/outputs")

executor.update_parameters(timeout_min = 60, mem_gb = 1, gpus_per_node = 1, cpus_per_task = 1, slurm_array_parallelism = 1, slurm_partition = "gpu")

jobs = []
with executor.batch():
	job = executor.submit(train, T=T, k=n, GAMMA=GAMMA, max_episode_num=episodes, max_steps=max_steps, lr=lr, experiment_path = run_path)
	jobs.append(job)
