# custom utilies for displaying animation, collecting rollouts and more
import pong_utils_stochastic
from parallelEnv import parallelEnv
import numpy as np
import torch.optim as optim
import gym
import os
import time
import torch

"""# check which device is being used.
# I recommend disabling gpu until you've made sure that the code runs
device = pong_utils_cleaner.device

policy = pong_utils_cleaner.Policy().to(device)
# we use the adam optimizer with learning rate 2e-4
# optim.SGD is also possible"""

def train(episode, R, r, n, tmax, experiment_path, folder_name, randrew = True, preagent = False, generalising=False, curriculum=False, save_model=False):
    device = pong_utils_stochastic.device

    policy = pong_utils_stochastic.Policy().to(device)

    #load pretrained agent
    pretrained_agent = None
    if preagent:
        """init_folder = 't75eps10000n16'
        path = os.path.join(init_folder, 'model.pt')
        init_path = os.path.join('pong_upgraded', '202304-0313-0328')
        final_path = os.path.join('results', init_path)
        actual_path = os.path.join(final_path, path)
        pretrained_agent = pong_utils_stochastic.Policy().to(device)
        pretrained_agent.load_state_dict(torch.load(actual_path, map_location=device))"""

        init_folder_1 = 't110eps15000n20trial0'
        path_1 = os.path.join(init_folder_1, 'model.pt')
        init_path_1 = os.path.join('pong_randreward_preagent_gen', '202304-1622-5321')
        final_path_1 = os.path.join('results', init_path_1)
        actual_path_1 = os.path.join(final_path_1, path_1)
        pretrained_agent_1 = pong_utils_stochastic.Policy().to(device)
        pretrained_agent_1.load_state_dict(torch.load(actual_path_1, map_location=device))

    # we use the adam optimizer with learning rate 2e-4
    # optim.SGD is also possible
    #import torch.optim as optim
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    # initialize environment
    envs = parallelEnv('PongDeterministic-v4', n=n, seed=1234, repeat_action_probability=0.2)

    path = os.path.join(experiment_path, folder_name)
    os.makedirs(path, exist_ok=True)
    #model_path = os.path.join(path, 'model.pt')
    file_path = os.path.join(path, 'dic.npy')
    model_path = os.path.join(path, 'model.pt')

    discount_rate = .99
    beta = .01


    dic = dict()
    # keep track of progress
    #dic['r'] = np.zeros(episode)
    #dic['t'] = np.zeros(episode)

    dic['r'] = np.zeros((episode, n))
    dic['t'] = np.zeros((episode, n))

    for e in range(episode):
        # collect trajectories
        old_probs, states, actions, rewards, rewards_mask, time_od, fr1, fr2 = \
            pong_utils_stochastic.collect_trajectories(envs, policy, R, r, randrew, tmax=tmax, preagent=pretrained_agent)

        if curriculum:
            if np.mean(rewards_mask) >= 0.8:
                tmax += 14

        total_rewards = np.sum(rewards, axis=0)

        # this is the SOLUTION!
        # use your own surrogate function
        # L = -surrogate(policy, old_probs, states, actions, rewards, beta=beta)

        L = -pong_utils_stochastic.surrogate(policy, old_probs, states, actions, rewards, beta=beta)
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        del L

        if generalising:
            #print(rewards_mask)
            while True:
                if not np.any(rewards_mask):
                    break
                batch_input = pong_utils_stochastic.preprocess_batch([fr1, fr2])
                # probs will only be used as the pi_old
                # no gradient propagation is needed
                # so we move it to the cpu
                probs = policy(batch_input).squeeze().cpu().detach().numpy()
                action = np.where(np.random.rand(n) < probs, 4, 5)
                # advance the game (0=no action)
                # we take one action and skip game forward
                fr1, re1, is_done1, _ = envs.step(action)
                fr2, re2, is_done2, _ = envs.step([0] * n)

                reward = re1 + re2
                is_done = np.logical_or(is_done1, is_done2)
                mask = np.where(reward < 0, 0, 1)
                rewards_mask *= mask
                time_od += rewards_mask
                if np.any(is_done):
                    print('We have a winner! (or loser)')
                    break

            #dic['t'][e] = np.mean(time_od)
            dic['t'][e,:] = time_od
        else:
            dic['t'][e,:] = time_od

        # the regulation term also reduces
        # this reduces exploration in later runs
        beta *= .995

        # get the average reward of the parallel environments

        #dic['r'][e] = (np.mean(total_rewards))
        dic['r'][e, :] = total_rewards

        # display some progress every 20 iterations
        if (e + 1) % 100 == 0:
            print(e)
            print(time_od)
            #print("Episode: {0:d}, score: {1:f}".format(e + 1, np.mean(total_rewards)))
            #print(total_rewards)
            np.save(file_path, dic)

            if (e + 1) % 1000 == 0:
                torch.save(policy.state_dict(), model_path)


    # update progress widget bar
    #timer.update(e + 1)
    if save_model:
        torch.save(policy.state_dict(), model_path)

    return 0
#timer.finish()