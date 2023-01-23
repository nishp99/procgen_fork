import gym
import numpy as np
import math
import os
import submitit
import datetime
import simple_training
import policy_network
#from policy_network import *
from pong_training import *
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

experiment_path = os.path.join(results_path, "pong_results")
os.makedirs(experiment_path, exist_ok = True)

run_path = os.path.join(experiment_path, run_timestamp)
os.mkdir(run_path)

episodes = 1000000
max_steps = 1000
lr = 1e-4
GAMMA = 0.99

executor = submitit.AutoExecutor(folder="utils/results/outputs")

executor.update_parameters(timeout_min = 10000, mem_gb = 3, gpus_per_node = 1, cpus_per_task = 1, slurm_array_parallelism = 256, slurm_partition = "gpu")

jobs = []

with executor.batch():
	job = executor.submit(train, GAMMA=GAMMA, max_episode_num=episodes, max_steps=max_steps, lr=lr, experiment_path=run_path, folder_name = 'ponglre_4gamma099')
	jobs.append(job)
