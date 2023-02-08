# custom utilies for displaying animation, collecting rollouts and more
import pong_utils
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

def train(episode, experiment_path, folder_name):
    device = pong_utils.device

    policy = pong_utils.Policy().to(device)

    # we use the adam optimizer with learning rate 2e-4
    # optim.SGD is also possible
    #import torch.optim as optim
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    # initialize environment
    envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)

    path = os.path.join(experiment_path, folder_name)
    os.makedirs(path, exist_ok=True)
    #model_path = os.path.join(path, 'model.pt')
    file_path = os.path.join(path, 'dic.npy')

    discount_rate = .99
    beta = .01
    tmax = 20

    # keep track of progress
    mean_rewards = np.zeros(episode)

    for e in range(episode):
        # collect trajectories
        old_probs, states, actions, rewards = \
            pong_utils.collect_trajectories(envs, policy, tmax=tmax)

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
        mean_rewards[e] = (np.mean(total_rewards))

        # display some progress every 20 iterations
        if (e + 1) % 20 == 0:
            print(e)
            #print("Episode: {0:d}, score: {1:f}".format(e + 1, np.mean(total_rewards)))
            #print(total_rewards)
            np.save(file_path, mean_rewards)

    # update progress widget bar
    #timer.update(e + 1)

#timer.finish()
