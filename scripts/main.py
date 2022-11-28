from training_func import train
T = 3
n = 2
GAMMA = 0.9
meta_episodes = 10
max_steps = 20
lr = 3e-4

rewards = train(T, n, GAMMA, meta_episodes, max_steps)

print(rewards)

