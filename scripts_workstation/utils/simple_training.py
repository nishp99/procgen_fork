import gym
# import policy_network
from scripts_workstation.utils.policy_network import ImpalaCNN
# import update
from scripts_workstation.utils.update import return_gradient
from scripts_workstation.utils.entropy_update import return_gradient_entropy
from scripts_workstation.utils.framestack import *
import numpy as np
from procgen import ProcgenEnv
import os


# import pdb

def train(T, k, GAMMA, max_episode_num, max_steps, lr, experiment_path):
    print('about to make leaper')
    # pdb.set_trace()
    env = gym.make("procgen:procgen-leaper-v0")
    env = FrameStack(env, 5)
    # print(env.observation_space)
    """env = ProcgenEnv(num_envs=1, env_name="leaper")
    env = VecExtractDictObs(env, "rgb")
    env = TransposeFrame(env)
    env = ScaledFloatFrame(env)"""
    print('made leaper')
    # env.render()
    policy_net = ImpalaCNN(env.observation_space, 2, lr)
    action_dict = {0: 5, 1: 4}
    # numsteps = []
    # avg_numsteps = []
    data = dict()
    data['rew'] = np.zeros(max_episode_num)
    data['eps'] = np.zeros(max_episode_num)
    frames = 5

    path = os.path.join(experiment_path, f'3_lane_simple')
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, 'dic.npy')

    for episode in range(max_episode_num):
        state = env.reset()
        log_probs = []
        probs = []
        rewards = []

        if episode % 10000 == 0:
            np.save(file_path, data)

        for steps in range(max_steps):
            # env.render()
            # action, log_prob = policy_net.get_action_log_prob(state)
            action, prob = policy_net.get_action_prob(state)
            action = action_dict[int(action.item())]
            for f in range(frames):
                new_state, reward, done, _ = env.step(action)
                if done:
                    probs.append(prob)
                    rewards.append(reward)
                    data['rew'][episode] = sum(rewards)
                    data['eps'][episode] = steps
                    return_gradient_entropy(rewards, probs, GAMMA)
                    policy_net.optimizer.step()
                    policy_net.optimizer.zero_grad()
                    break
            if done:
                break
            # new_state, reward, done, _ = env.step(action)
            # log_probs.append(log_prob)
            probs.append(prob)
            rewards.append(reward)
            state = new_state

    np.save(file_path, data)

import datetime
import os
import numpy as np

T = 2
k = 1
GAMMA = 1
episodes = 300000*T
max_steps = 100
lr = 1e-5

run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
results_path = os.path.join("scripts_workstation", "utils", "results")
os.makedirs(results_path, exist_ok = True)
experiment_path = os.path.join(results_path, "test")
os.makedirs(experiment_path, exist_ok = True)
print(os.getcwd())
run_path = os.path.join(experiment_path, run_timestamp)

rewards = train(T, k, GAMMA, episodes, max_steps, lr, run_path)
