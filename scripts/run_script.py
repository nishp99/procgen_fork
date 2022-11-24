import gym

env = gym.make('procgen:procgen-leaper-v0')
rewards, dones = [], []

state = env.reset()

for i in range(30):
	new_state, reward, done, _ = env.step(5)
	rewards.append(reward)
	dones.append(done)

print((rewards,dones))
