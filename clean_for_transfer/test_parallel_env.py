import numpy as np
import gym
import random as rand
from parallelEnv_changed import parallelEnv

envs = parallelEnv('LunarLander-v2', n=4)