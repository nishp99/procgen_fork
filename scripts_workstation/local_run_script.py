from utils.training_func import train
import datetime
import os
import numpy as np

T = 2
k = 1
GAMMA = 1
episodes = 4000000*T
max_steps = 100
lr = 5e-6

run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
results_path = os.path.join("scripts_workstation", "utils", "results")
os.makedirs(results_path, exist_ok = True)
experiment_path = os.path.join(results_path, "test")
os.makedirs(experiment_path, exist_ok = True)
print(os.getcwd())
run_path = os.path.join(experiment_path, run_timestamp)

rewards = train(T, k, GAMMA, episodes, max_steps, lr, run_path)

#start timestamp with unique identifier for name
# run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
#os.mkdir(with name of unique identifier)

#os.path.join(results, unique identifier)
# results_path = os.path.join("utils", "results")
# os.makedirs(results_path, exist_ok = True)