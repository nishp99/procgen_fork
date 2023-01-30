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

experiment_path = os.path.join(results_path, "new_pong_results")
os.makedirs(experiment_path, exist_ok = True)

run_path = os.path.join(experiment_path, run_timestamp)
os.mkdir(run_path)

episodes = 1000000
max_steps = 500
lr = 1e-4
GAMMA = 0.995
n_s = [1,2,3,4,5]

executor = submitit.AutoExecutor(folder="utils/results/pong_results")

executor.update_parameters(timeout_min = 10000, mem_gb = 3, gpus_per_node = 1, cpus_per_task = 1, slurm_array_parallelism = 256, slurm_partition = "gpu")

jobs = []

with executor.batch():
	for n in n_s:
		for i in range(6):
			job = executor.submit(train, max_steps=max_steps, lr=lr, experiment_path=run_path, folder_name = f'pongn_{n}{i}gamma0995', n = n,  max_episode_num=episodes, opp_rew = 1, win_reward = 2, GAMMA= GAMMA)
			jobs.append(job)
