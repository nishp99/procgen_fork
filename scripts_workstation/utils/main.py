from training_func import train
T = 3
n = 2
GAMMA = 0.9
episodes = 4*T
max_steps = 60
lr = 3e-4

rewards = train(T, n, GAMMA, episodes, max_steps, lr, 'path')

print(rewards)

