import gym
from policy_network import PolicyNetwork
from update import return_gradient
import numpy as np
from procgen import ProcgenEnv

def train(T,k, GAMMA, max_episode_num, max_steps):
    env = gym.make("procgen:procgen-leaper-v0")
    obs = env.reset()
    tobs = env.reset()
    #env.render()
    policy_net = PolicyNetwork(env.observation_space, 2)
    action_dict = {0:4, 1:5}
    #numsteps = []
    #avg_numsteps = []
    all_rewards = []
    t = 0
    lives = k

    for episode in range(max_episode_num):
        state = env.reset()
        log_probs = []
        rewards = []

        for steps in range(max_steps):
            #env.render()
            action, log_prob = policy_net.get_action(state)
            action = action_dict[action]
            new_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                t += 1
                if reward:
                    if t%T == 0:
                        all_rewards.append(np.sum(rewards))
                        return_gradient(rewards, log_probs, GAMMA)
                        policy_net.optimizer.step()
                        policy_net.optimizer.zero_grad()
                        t = 0
                        lives = k
                        break
                    else:
                        all_rewards.append(np.sum(rewards))
                        return_gradient(rewards, log_probs, GAMMA)
                        break
                else:
                    if lives == 1:
                        t = 0
                        lives = k
                        all_rewards.append(np.sum(rewards))
                        policy_net.optimizer.zero_grad()
                    else:
                        lives -= 1
                        return_gradient(rewards, log_probs, GAMMA)
                        all_rewards.append(np.sum(rewards))
                        break


            """if done: #if reward==1, and if t%T == 0
                update_policy(policy_net, rewards, log_probs)
                numsteps.append(steps)
                avg_numsteps.append(np.mean(numsteps[-10:]))
                all_rewards.append(np.sum(rewards))
                if episode % 1 == 0:
                    sys.stdout.write("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode,np.round(np.sum(rewards),decimals=3),np.round(np.mean(all_rewards[-10:]),decimals=3),steps))
                break"""

            state = new_state

    """plt.plot(numsteps)
    plt.plot(avg_numsteps)
    plt.xlabel('Episode')
    plt.show()"""
    return all_rewards