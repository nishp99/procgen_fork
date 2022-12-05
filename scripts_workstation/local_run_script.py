from utils.training_func import train
import datetime
import os

T = 2
n = 1
GAMMA = 0.99
episodes = 10000*T
max_steps = 100
lr = 1e-4

run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
results_path = os.path.join("utils", "results")
os.makedirs(results_path, exist_ok = True)
experiment_path = os.path.join(results_path, "test")
os.makedirs(experiment_path, exist_ok = True)
run_path = os.path.join(experiment_path, run_timestamp)
os.mkdir(run_path)

#start timestamp with unique identifier for name
# run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
#os.mkdir(with name of unique identifier)

#os.path.join(results, unique identifier)
# results_path = os.path.join("utils", "results")
# os.makedirs(results_path, exist_ok = True)

#rewards = train(T, n, GAMMA, episodes, max_steps, lr, run_path)

print(rewards[5000:])
