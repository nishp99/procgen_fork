import torch
import numpy as np

def return_gradient_entropy(rewards, probs, GAMMA):
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
    """discounted_rewards[0] = 1
    for i in range(1, len(rewards)):
        discounted_rewards[i] = discounted_rewards[i] * discounted_rewards[i - 1]
    discounted_rewards = discounted_rewards[::-1]"""
    discounted_rewards = torch.from_numpy(discounted_rewards.copy())
    #discounted_rewards = rewards[-1]*discounted_rewards
    discounted_rewards = rewards[-1]*discounted_rewards
    #discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                #discounted_rewards.std() + 1e-9)  # normalize discounted rewards

    policy_gradient = []
    for prob, Gt in zip(probs, discounted_rewards):
        log_prob = torch.log(prob)
        log_prob_neg = torch.log(1-prob)
        policy_gradient.append(-log_prob * Gt + 0.1*(prob*log_prob + (1-prob)*log_prob_neg))

    #policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    #policy_network.optimizer.step()