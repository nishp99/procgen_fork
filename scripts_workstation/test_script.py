import gym
from utils.framestack import *
from procgen import ProcgenEnv
import gym.wrappers
"""n_envs = 2

print("about to make leaper")
env = gym.make("procgen:procgen-leaper-v0", render_mode = "rgb_array")
#env = FrameStack(env,5)
obs = env.reset()

action = 4

n = 0

while True:
    obs, reward, done, _ = env.step(action)
    env.render()
    n += 1
    if n >= 150:
        break
    if done:
        break"""


env = gym.make("procgen:procgen-coinrun-v0", render_mode = "rgb_array")
env.metadata["render.modes"] = ["human", "rgb_array"]

env = gym.wrappers.Monitor(env=env, directory="./videos", force = True)

episodes = 1
_ = env.reset()

done = False
while episodes > 0:
    _, _, done, _ = env.step(env.action_space.sample())
    if done:
        _ = env.reset()
        episodes -= 1