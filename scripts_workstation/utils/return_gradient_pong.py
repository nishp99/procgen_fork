import torch
import numpy as np

def return_gradient_entropy(rewards, probs, GAMMA=0.999, device):
    #discounted_rewards = np.zeros(len(rewards))
    #discounted_rewards = np.zeros(nums_envs, max_steps)
    rewards = (torch.from_numpy(rewards.copy())).to(device)
    n = rewards.shape[1]
    step = torch.arange(n)[:, None] - torch.arange(n)[None, :]
    ones = torch.ones_like(step)
    zeros = torch.zeros_like(step)

    target = torch.where(step >= 0, ones, zeros)
    step = torch.where(step >= 0, step, zeros)
    discount = target * (GAMMA ** step)
    discount = discount.to(device)

    discounted_rewards = torch.mm(rewards, discount)

    """for t in range(len(rewards)): #in range(max_steps)
        Gt = 0 #np.zeros(num_envs)
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA ** pw * r
            pw = pw + 1
        discounted_rewards[t] = Gt"""


    #discounted_rewards = (torch.from_numpy(discounted_rewards.copy())).to(device)
    #discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std())  # normalize discounted rewards
    discounted_rewards = (discounted_rewards - torch.mean(discounted_rewards, axis=1)) / (torch.std(discounted_rewards, axis=1)+1e-10)

    probabilities = (torch.from_numpy(probs.copy())).to(device)
    log_probabilities = torch.log(probabilities + 1e-10)
    policy_gradient = - log_probabilities * discounted_rewards



    #policy_network.optimizer.zero_grad()
    policy_gradient = torch.mean(policy_gradient)#= torch.sum(policy_gradient)/num_envs
    policy_gradient.backward()
    #policy_network.optimizer.step()