import gym
#import policy_network
#from policy_network import ImpalaCNN
#import update
#from update import return_gradient
import numpy as np
from procgen import ProcgenEnv
import os
import pdb

def train(T,k, GAMMA, max_episode_num, max_steps, lr, experiment_path):
    print('about to make leaper')
    pdb.set_trace()
    env = gym.make("procgen:procgen-leaper-v0")
    print('made leaper')
    #env.render()
    policy_net = ImpalaCNN(env.observation_space, 2, lr)
    action_dict = {0:4, 1:5}
    #numsteps = []
    #avg_numsteps = []
    data = dict()
    data['rew'] = np.zeros(max_episode_num)
    data['eps'] = np.zeros(max_episode_num)
    t = 0
    lives = k

    for episode in range(max_episode_num):
        state = env.reset()
        log_probs = []
        rewards = []

        for steps in range(max_steps):
            #env.render()
            action, log_prob = policy_net.get_action(state)
            action = action_dict[int(action.item())]
            new_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                t += 1
                if reward:
                    if t%T == 0:
                        data['rew'][episode] = np.sum(rewards)
                        data['eps'][episode] = steps
                        return_gradient(rewards, log_probs, GAMMA)
                        policy_net.optimizer.step()
                        policy_net.optimizer.zero_grad()
                        t = 0
                        lives = k
                        break
                    else:
                        data['rew'][episode] = np.sum(rewards)
                        data['eps'][episode] = steps
                        return_gradient(rewards, log_probs, GAMMA)
                        break
                else:
                    if lives == 1:
                        t = 0
                        lives = k
                        data['rew'][episode] = np.sum(rewards)
                        data['eps'][episode] = steps
                        policy_net.optimizer.zero_grad()
                        break
                    elif t%T == 0:
                        data['rew'][episode] = np.sum(rewards)
                        data['eps'][episode] = steps
                        return_gradient(rewards, log_probs, GAMMA)
                        policy_net.optimizer.step()
                        policy_net.optimizer.zero_grad()
                        t = 0
                        lives = k
                        break
                    else:
                        lives -= 1
                        return_gradient(rewards, log_probs, GAMMA)
                        data['rew'][episode] = np.sum(rewards)
                        data['eps'][episode] = steps
                        break

            state = new_state

    path = os.path.join(experiment_path, f'{T}-{k}-{GAMMA}')
    os.mkdir(path)
    file_path = os.path.join(path, 'dic.npy')
    np.save(file_path, data)
