import gym
import numpy as np
import math
import os
import submitit
import datetime
import simple_training
import policy_network
#from policy_network import *
from simple_training import *
#from procgen import ProcgenEnv
#import update
import entropy_update
#from update import *
#from entropy_update import *
import sys
import os
#sys.path.append('utils')

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
#T = 4
#n = 3
GAMMA = 1
episodes = 1000000
max_steps = 100
lr = 1e-5

executor = submitit.AutoExecutor(folder="utils/results/outputs")

executor.update_parameters(timeout_min = 840, mem_gb = 3, gpus_per_node = 1, cpus_per_task = 1, slurm_array_parallelism = 1, slurm_partition = "gpu")

jobs = []
with executor.batch():
	job = executor.submit(train, GAMMA=GAMMA, max_episode_num=episodes, max_steps=max_steps, lr=lr, experiment_path = run_path, full_actions = True, use_entropy = True)
	jobs.append(job)
