"""def update_policy(policy_network, rewards, log_probs):
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA ** pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)

    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + 1e-9)  # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)

    #policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()"""

import torch
import numpy as np

def return_gradient(rewards, log_probs, GAMMA):
    discounted_rewards = GAMMA*np.ones(len(rewards)) #may have to keep previous formulation, as otherwise get zeros later on
    #there is motivation to remove discount, as would place relatively high value on incorrect moves (when lives remain)
    """
    discounted_rewards = len(rewards)*[rewards[-1]]
    """
    """for t in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA ** pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)"""

    #discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards[0] = 1
    for i in range(1, len(rewards)):
        discounted_rewards[i] = discounted_rewards[i] * discounted_rewards[i - 1]
    discounted_rewards = discounted_rewards[::-1]
    discounted_rewards = torch.from_numpy(discounted_rewards.copy())
    discounted_rewards = rewards[-1]*discounted_rewards
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + 1e-9)  # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)

    #policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    #policy_network.optimizer.step()