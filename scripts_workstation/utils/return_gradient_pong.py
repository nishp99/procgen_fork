import torch
import numpy as np

def return_gradient_entropy(rewards, log_probs, GAMMA, device):
    discounted_rewards = np.zeros(len(rewards))

    for t in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA ** pw * r
            pw = pw + 1
        discounted_rewards[t] = Gt

    discounted_rewards = (torch.from_numpy(discounted_rewards.copy())).to(device)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std())  # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)

    #policy_network.optimizer.zero_grad()
    policy_gradient = (torch.stack(policy_gradient)).sum()
    policy_gradient.backward()
    #policy_network.optimizer.step()