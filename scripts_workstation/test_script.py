import gym
from utils.framestack import *
from procgen import ProcgenEnv
n_envs = 2

print("about to make leaper")
env = gym.make("procgen:procgen-leaper-v0", render_mode = "rgb_array")
#env = FrameStack(env,5)
env.reset()
a = 1
for i in range(30):
    if i%4 < 2:
        add = 3
    else:
        add = -3
    for z in range(5):
        obs, rew, done, info = env.step(a)
        #env.render()
    a += add
# env = ProcgenEnv(num_envs=n_envs, env_name="leaper")
#env = VecExtractDictObs(env, "rgb")
#env = TransposeFrame(env)
#env = ScaledFloatFrame(env)
#env = gym.make("procgen:procgen-leaper-v0")
#print(env.observation_space)
print("made leaper")