import numpy as np
import gym
import random as rand
from parallelEnv_changed import parallelEnv

envs = parallelEnv('PongDeterministic-v4', n=4)