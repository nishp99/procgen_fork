import gym
# import policy_network
#from scripts_workstation.utils.policy_network import ImpalaCNN
from policy_network import ImpalaCNN
#from scripts_workstation.utils.new_network import NatureModel
# import update
#from scripts_workstation.utils.update import return_gradient
#from scripts_workstation.utils.entropy_update import return_gradient_entropy
from entropy_update import return_gradient_entropy
#from scripts_workstation.utils.framestack import *
from framestack import *
import numpy as np
#from procgen import ProcgenEnv
import os
import torch


# import pdb

def train(GAMMA, max_episode_num, max_steps, lr, experiment_path, full_actions, use_entropy):
    gpu = torch.cuda.get_device_name(0)
    print(f'gpu:{gpu}')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(f'is gpu available: {torch.cuda.is_available()}')
    print(f'device count: {torch.cuda.device_count()}')
    print('about to make leaper')

    env = gym.make("procgen:procgen-leaper-v0")
    env = FrameStack(env, 5)

    print('made leaper')
    if full_actions:
        num_actions = 5
    else:
        num_actions = 2

    policy_net = ImpalaCNN(env.observation_space, num_actions, lr)
    #policy_net = NatureModel(env.observation_space, 2, lr)
    policy_net.to(device)
    action_dict = {0:4, 1:5, 2:3, 3:1, 4:7}
    use_entropy = int(use_entropy)

    data = dict()
    data['rew'] = np.zeros(max_episode_num)
    data['eps'] = np.zeros(max_episode_num)
    frames = 5

    path = os.path.join(experiment_path, f'3_lane_simple')
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, 'dic.npy')

    for episode in range(max_episode_num):
        state = env.reset()
        log_probs_entropies = []
        rewards = []

        if episode % 10000 == 0:
            np.save(file_path, data)

        for steps in range(max_steps):
            action, log_prob, entropy = policy_net.get_action(state, device)
            action = action_dict[int(action.item())]
            for f in range(frames):
                new_state, reward, done, _ = env.step(action)
                if done:
                    log_probs_entropies.append((log_prob, entropy))
                    rewards.append(reward)
                    data['rew'][episode] = sum(rewards)
                    data['eps'][episode] = steps
                    policy_net.optimizer.zero_grad()
                    return_gradient_entropy(rewards, log_probs_entropies, GAMMA, device, use_entropy)
                    policy_net.optimizer.step()
                    break
            if done:
                break
            log_probs_entropies.append((log_prob, entropy))
            rewards.append(reward)
            state = new_state

    np.save(file_path, data)

"""import datetime
import os
import numpy as np

GAMMA = 1
episodes = 300000
max_steps = 100
lr = 1e-5

run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
results_path = os.path.join("scripts_workstation", "utils", "results")
os.makedirs(results_path, exist_ok = True)
experiment_path = os.path.join(results_path, "test")
os.makedirs(experiment_path, exist_ok = True)
print(os.getcwd())
run_path = os.path.join(experiment_path, run_timestamp)

rewards = train(GAMMA, episodes, max_steps, lr, run_path)"""
