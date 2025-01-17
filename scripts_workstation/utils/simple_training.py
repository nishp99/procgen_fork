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

def train(GAMMA, max_episode_num, max_steps, lr, experiment_path, num_actions, entropy_factor, folder_name, game, zero_rewards, zero_observations):
    print(torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(f'is gpu available: {torch.cuda.is_available()}')
    print(f'device count: {torch.cuda.device_count()}')
    #gpu = torch.cuda.get_device_name(0)
    #print(f'gpu:{gpu}')
    print('about to make leaper')

    game_action_dict = {'leaper': {0:4, 1:5, 2:3, 3:1, 4:7}, 'bigfish': {0:5, 1:3, 2:4, 3:1, 4:7}}
    frame_dict = {'leaper': 5, 'bigfish': 4}

    action_dict = game_action_dict[game]
    frames = frame_dict[game]
    #use_entropy = int(use_entropy)

    procgen_dict = {'bigfish': "procgen:procgen-bigfish-v0", 'leaper': "procgen:procgen-leaper-v0"}
    procgen_game = procgen_dict[game]
    env = gym.make(procgen_game)
    env = FrameStack(env, frames)

    print('made leaper')
    #model_path = os.path.join(experiment_path, 'model.pt')

    policy_net = ImpalaCNN(env.observation_space, num_actions, lr)
    policy_net.to(device)

    data = dict()
    data['rew'] = np.zeros(max_episode_num)
    data['eps'] = np.zeros(max_episode_num)

    path = os.path.join(experiment_path, folder_name)
    os.makedirs(path, exist_ok=True)
    model_path = os.path.join(path, 'model.pt')
    file_path = os.path.join(path, 'dic.npy')

    for episode in range(max_episode_num):
        state = env.reset()
        state = np.array(state)
        state = state / 255.0
        if zero_observations:
            state = (state - np.mean(state, axis=(1,2), keepdims=True))/np.var(state, axis=(1,2), keepdims=True)

        log_probs_entropies = []
        rewards = []

        if episode % 10000 == 0:
            np.save(file_path, data)
            if episode % 50000 == 0:
                torch.save(policy_net.state_dict(), model_path)

        for steps in range(max_steps):
            action, log_prob, entropy = policy_net.get_action(state, device, frames)
            action = action_dict[int(action.item())]
            for f in range(frames):
                new_state, reward, done, _ = env.step(action)
                if done:
                    log_probs_entropies.append((log_prob, entropy))
                    rewards.append(reward)
                    data['rew'][episode] = sum(rewards)
                    data['eps'][episode] = steps
                    policy_net.optimizer.zero_grad()
                    return_gradient_entropy(rewards, log_probs_entropies, GAMMA, device, entropy_factor, zero_rewards)
                    policy_net.optimizer.step()
                    break
            if done:
                break
            log_probs_entropies.append((log_prob, entropy))
            rewards.append(reward)
            new_state = np.array(new_state)
            state = new_state/255.0
            if zero_observations:
                state = (state - np.mean(state, axis=(1, 2), keepdims=True))/np.var(state, axis=(1,2), keepdims=True)

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
