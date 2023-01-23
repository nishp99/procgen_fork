import gym
# import policy_network
#from scripts_workstation.utils.policy_network import ImpalaCNN
from pong_policy import Policy
#from scripts_workstation.utils.new_network import NatureModel
# import update
#from scripts_workstation.utils.update import return_gradient
#from scripts_workstation.utils.entropy_update import return_gradient_entropy
from return_gradient_pong import return_gradient_entropy
#from scripts_workstation.utils.framestack import *
from framestack import *
import numpy as np
#from procgen import ProcgenEnv
import os
import torch


# import pdb

def train(max_episode_num = 1000, max_steps, lr, experiment_path, folder_name, GAMMA=0.99):
    print(torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(f'is gpu available: {torch.cuda.is_available()}')
    print(f'device count: {torch.cuda.device_count()}')
    #gpu = torch.cuda.get_device_name(0)
    #print(f'gpu:{gpu}')

    bkg_color = np.array([144, 72, 17])
    def prepro(image):
        img = np.mean(image[34:-16:2, ::2] - bkg_color, axis=-1) / 255.
        return img

    env = gym.make('Pong-v4')
    #env = FrameStack(env, frames)

    print('made pong')
    #model_path = os.path.join(experiment_path, 'model.pt')

    policy_net = Policy(lr)
    policy_net.to(device)

    data = dict()
    data['rew'] = np.zeros(max_episode_num)
    data['eps'] = np.zeros(max_episode_num)

    path = os.path.join(experiment_path, folder_name)
    os.makedirs(path, exist_ok=True)
    model_path = os.path.join(path, 'model.pt')
    file_path = os.path.join(path, 'dic.npy')

    for episode in range(max_episode_num):
        cur_s = env.reset()
        cur_s = prepro(cur_s)
        prev_s = None

        rewards = []
        log_probs = []

        if episode % 10000 == 0:
            np.save(file_path, data)
            if episode % 50000 == 0:
                torch.save(policy_net.state_dict(), model_path)

        for steps in range(max_steps):
            x = cur_s - prev_s if prev_s is not None else np.zeros_like(cur_s)
            prev_s = cur_s
            action, log_prob = policy_net.forward(x, device)
            new_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            if done:
                data['rew'][episode] = sum(rewards)
                data['eps'][episode] = steps
                policy_net.optimizer.zero_grad()
                return_gradient_entropy(rewards, log_probs, GAMMA, device)
                policy_net.optimizer.step()
                break
            cur_s = prepro(new_state)

    np.save(file_path, data)
    torch.save(policy_net.state_dict(), model_path)

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
