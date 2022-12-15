import gym
from utils.framestack import *
from procgen import ProcgenEnv
n_envs = 2

print("about to make leaper")
env = gym.make("procgen:procgen-leaper-v0")
# env = ProcgenEnv(num_envs=n_envs, env_name="leaper")
#env = VecExtractDictObs(env, "rgb")
#env = TransposeFrame(env)
#env = ScaledFloatFrame(env)
#env = gym.make("procgen:procgen-leaper-v0")
#print(env.observation_space)
print("made leaper")
