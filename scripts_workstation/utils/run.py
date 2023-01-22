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

episodes = 10000000
max_steps = 300
lr = 1e-5

executor = submitit.AutoExecutor(folder="utils/results/outputs")

executor.update_parameters(timeout_min = 10000, mem_gb = 3, gpus_per_node = 1, cpus_per_task = 1, slurm_array_parallelism = 256, slurm_partition = "gpu")

jobs = []

#entropy_factors = [0, 0.01, 0.1, 0.6, 1, 10, 100]
entropy_factors = [0.1]
entropy_names = {0:'0', 0.01:'001', 0.1:'01', 0.6:'06', 1:'1', 10:'10', 100:'100'}

GAMMAS = [0.9]
GAMMA_names = {0.9:'09'}

#games = ['leaper', 'bigfish']
games = ['leaper']
game_folder_name = {'leaper': 'leaper4lane', 'bigfish': 'bigfish3fish'}

#game_actions = ['reduced', 'all']
game_actions = ['all']
action_numbers = {'leaper': {'reduced': 2, 'all': 5}, 'bigfish': {'reduced': 3, 'all': 5}}

zero_rewards = False
zero_observations = False

mean_rewards = {True: 'MeanRew', False: ''}
mean_obs = {True: 'MeanObs', False: ''}

with executor.batch():
	for GAMMA in GAMMAS:
		for entropy_factor in entropy_factors:
			for game in games:
				for actions in game_actions:
					folder = f'{game_folder_name[game]}{actions}actent{entropy_names[entropy_factor]}gamma{GAMMA_names[GAMMA]}{mean_rewards[zero_rewards]}{mean_obs[zero_observations]}'
					job = executor.submit(train, GAMMA=GAMMA, max_episode_num=episodes, max_steps=max_steps, lr=lr, experiment_path=run_path, num_actions=action_numbers[game][actions], entropy_factor=entropy_factor, folder_name=folder, game=game, zero_rewards=zero_rewards, zero_observations=zero_observations)
					jobs.append(job)
