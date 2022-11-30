from utils.training_func import train
T = 2
n = 1
GAMMA = 0.99
episodes = 300*T
max_steps = 100
lr = 9e-4

rewards = train(T, n, GAMMA, episodes, max_steps, lr, 'path')

print(rewards)
