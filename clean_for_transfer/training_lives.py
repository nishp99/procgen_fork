# custom utilies for displaying animation, collecting rollouts and more
import pong_utils_lives
from parallelEnv import parallelEnv
import numpy as np
import torch.optim as optim
import gym
import os
import time

# check which device is being used.
# I recommend disabling gpu until you've made sure that the code runs
device = pong_utils.device

policy=pong_utils.Policy().to(device)

# we use the adam optimizer with learning rate 2e-4
# optim.SGD is also possible

def train(episode, R, n, k, tmax, experiment_path, folder_name, generalising = False):
    device = pong_utils.device

    policy = pong_utils.Policy().to(device)

    # we use the adam optimizer with learning rate 2e-4
    # optim.SGD is also possible
    #import torch.optim as optim
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    # initialize environment
    envs = parallelEnv('PongDeterministic-v4', n=n, seed=1234)

    path = os.path.join(experiment_path, folder_name)
    os.makedirs(path, exist_ok=True)
    #model_path = os.path.join(path, 'model.pt')
    file_path = os.path.join(path, 'dic.npy')

    discount_rate = .99
    beta = .01


    dic = dict()
    # keep track of progress
    dic['r'] = np.zeros((episode, n))
    dic['t'] = np.zeros((episode, n))

    for e in range(episode):
        # collect trajectories
        old_probs, states, actions, rewards, max_time = \
            pong_utils.collect_trajectories(envs, policy, R, k, tmax=tmax)

        total_rewards = np.sum(rewards, axis=0)

        # this is the SOLUTION!
        # use your own surrogate function
        # L = -surrogate(policy, old_probs, states, actions, rewards, beta=beta)

        L = -pong_utils.surrogate(policy, old_probs, states, actions, rewards, beta=beta)
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        del L

        # the regulation term also reduces
        # this reduces exploration in later runs
        beta *= .995

        # get the average reward of the parallel environments

        dic['r'][e, :] = total_rewards
        dic['t'][e, :] = max_time

        # display some progress every 20 iterations
        if (e + 1) % 100 == 0:
            print(e)
            #print("Episode: {0:d}, score: {1:f}".format(e + 1, np.mean(total_rewards)))
            #print(total_rewards)
            np.save(file_path, dic)

    # update progress widget bar
    #timer.update(e + 1)

#timer.finish()
