import gym
import torch
from policy_network import ImpalaCNN
from framestack import *
import os

game = 'bigfish'
num_actions = 2
date_time = "202301-0920-4817"

lr = 1e-5
init_path = os.path.join("scripts_workstation", "utils")
results_path = os.path.join("utils", "results")
experiment_path = os.path.join(results_path, "n_or_more")
path = os.path.join(experiment_path, date_time)
full_path = os.path.join(path, 'model.pt')
final_path = os.path.join(init_path, full_path)

game_action_dict = {'leaper': {0:4, 1:5, 2:3, 3:1, 4:7}, 'bigfish': {0:5, 1:3, 2:4, 3:1, 4:7}}
frame_dict = {'leaper': 5, 'bigfish': 4}

action_dict = game_action_dict[game]
frames = frame_dict[game]

procgen_dict = {'bigfish': "procgen:procgen-bigfish-v0", 'leaper': "procgen:procgen-leaper-v0"}
procgen_game = procgen_dict[game]
env = gym.make(procgen_game)
env = FrameStack(env, frames)

print(torch.cuda.device_count())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(f'is gpu available: {torch.cuda.is_available()}')
print(f'device count: {torch.cuda.device_count()}')
#gpu = torch.cuda.get_device_name(0)
#print(f'gpu:{gpu}')

policy_net = ImpalaCNN(env.observation_space, num_actions, lr)
policy_net.to(device)
policy_net.load_state_dict(torch.load(final_path, map_location=device))


for episode in range(100):
    state = env.reset()

    for steps in range(40):
        action, log_prob, entropy = policy_net.get_action(state, device, frames)
        action = action_dict[int(action.item())]
        print((action, torch.exp(log_prob)))

        for f in range(frames):
            new_state, reward, done, _ = env.step(action)
            #env.render()
            if done:
                break
        if done:
            break
        state = new_state






