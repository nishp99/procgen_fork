import gym
import torch
from policy_network import ImpalaCNN
from framestack import *
import os

game = 'leaper'
num_actions = 2
date_time = "place_holder"

lr = 1e-5
results_path = os.path.join("utils", "results")
experiment_path = os.path.join(results_path, "n_or_more")
path = os.path.join(experiment_path, date_time)
full_path = os.path.join(path, 'model.pt')

game_action_dict = {'leaper': {0:4, 1:5, 2:3, 3:1, 4:7}, 'bigfish': {0:5, 1:3, 2:4, 3:1, 4:7}}
frame_dict = {'leaper': 5, 'bigfish': 4}

action_dict = game_action_dict[game]
frames = frame_dict[game]

procgen_dict = {'bigfish': "procgen:procgen-bigfish-v0", 'leaper': "procgen:procgen-leaper-v0"}
procgen_game = procgen_dict[game]
env = gym.make(procgen_game)
env = FrameStack(env, frames)

policy_net = ImpalaCNN(env.observation_space, num_actions, lr)
policy_net.load_state_dict(torch.load(full_path))

for episode in range(max_episode_num):
    state = env.reset()

    for steps in range(max_steps):
        action, log_prob, entropy = policy_net.get_action(state, device, frames)
        action = action_dict[int(action.item())]

        for f in range(frames):
            new_state, reward, done, _ = env.step(action)
            env.render()
            if done:
                break
        if done:
            break
        state = new_state






