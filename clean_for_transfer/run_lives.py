import gym
import numpy as np
import math
import os
import submitit
import datetime

from training_lives import train
#rom procgen import ProcgenEnv
#import update
#import entropy_update
#from update import *
#from entropy_update import *
import sys
import os
#sys.path.append('utils')

#start timestamp with unique identifier for name
run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
#os.mkdir(with name of unique identifier)

#os.path.join(results, unique identifier)
results_path = os.path.join("results", "pong_lives")
os.makedirs(results_path, exist_ok = True)

experiment_path = os.path.join(results_path, "outputs")
os.makedirs(experiment_path, exist_ok = True)

run_path = os.path.join(results_path, run_timestamp)
os.mkdir(run_path)

executor = submitit.AutoExecutor(folder="results/pong_lives/outputs")

executor.update_parameters(timeout_min = 7000, mem_gb = 7, gpus_per_node = 1, cpus_per_task = 1, slurm_array_parallelism = 30, slurm_partition = "gpu")

jobs = []
lives = [1, 2, 3]
times = [50, 65, 80, 95, 110, 125, 140, 155, 170, 185]
episode = 20000
n = 16

with executor.batch():
    for k in lives:
        for tmax in times:
            job = executor.submit(train, episode=episode, R = 2, n = n, k = k, tmax = tmax, experiment_path=run_path, folder_name = f'lives{k}t{tmax}eps{episode}n{16}')
            jobs.append(job)
