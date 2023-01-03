import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module

class NatureModel(nn.Module):
    def __init__(self, obs_space, in_channels, learning_rate, **kwargs):
        """
        input_shape:  (tuple) tuple of the input dimension shape (channel, height, width)
        filters:       (list) list of the tuples consists of (number of channels, kernel size, and strides)
        use_batchnorm: (bool) whether to use batchnorm
        """
        super(NatureModel, self).__init__()
        f, h, w, c = obs_space.shape
        shape = (f * c, h, w)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels[0], out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=64*7*7, out_features=512), nn.ReLU()
        )
        self.output_dim = 512
        self.apply(orthogonal_init)
        self.fc_policy = orthogonal_init(nn.Linear(self.output_dim, 2), gain=0.01)
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    def forward(self):
        x = obs / 255.0  # scale to 0-1
        x = x.permute(3, 0, 1, 2)  # FHWC => CFHW
        x = x.reshape([15, 64, 64])
        x = self.layers(x)
        logits = self.fc_policy(x)
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)
        return p

    def get_action_log_prob(self, obs):
        #print(obs.shape)
        obs_np = np.array(obs)
        obs_tensor = torch.from_numpy(obs_np)
        dist = self.forward(obs_tensor)
        action = dist.sample()
        #print(dist.probs)
        return action, dist.log_prob(action)

    def get_action_prob(self, obs):
        obs_np = np.array(obs)
        obs_tensor = torch.from_numpy(obs_np)
        dist = self.forward(obs_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        #print(dist.probs)
        return action, torch.exp(log_prob)