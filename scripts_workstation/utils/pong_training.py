import gym
# import policy_network
#from scripts_workstation.utils.policy_network import ImpalaCNN
from pong_policy import Policy
#from scripts_workstation.utils.new_network import NatureModel
# import update
#from scripts_workstation.utils.update import return_gradient
#from scripts_workstation.utils.entropy_update import return_gradient_entropy
from return_gradient_pong import return_gradient_entropy
#from scripts_workstation.utils.framestack import *
from framestack import *
import numpy as np
#from procgen import ProcgenEnv
import os
import torch


# import pdb

def train(max_steps, lr, experiment_path, folder_name, n, max_episode_num, opp_rew, win_reward, GAMMA):
    num_envs = 8
    print(torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(f'is gpu available: {torch.cuda.is_available()}')
    print(f'device count: {torch.cuda.device_count()}')
    #gpu = torch.cuda.get_device_name(0)
    #print(f'gpu:{gpu}')
    action_dict = {0:2, 1:3}
    UP = 2
    DOWN = 3

    bkg_color = np.array([144, 72, 17])
    def prepro(image):
        img = np.mean(image[34:-16:2, ::2] - bkg_color, axis=-1) / 255.
        return img

    def preprocess_batch(images):
        list_of_images = np.asarray(images)
        if len(list_of_images.shape) < 5:
            list_of_images = np.expand_dims(list_of_images, 1)
        # subtract bkg and crop
        list_of_images_prepro = np.mean(list_of_images[:, :, 34:-16:2, ::2] - bkg_color,
                                        axis=-1) / 255.
        batch_input = np.swapaxes(list_of_images_prepro, 0, 1)
        return torch.from_numpy(batch_input).float().to(device)

    envs = gym.vector.make('Pong-v4', num_envs)


    print('made pong')
    #model_path = os.path.join(experiment_path, 'model.pt')

    policy_net = Policy(lr)
    policy_net.to(device)

    data = dict()
    data['rew'] = np.zeros(max_episode_num)
    data['eps'] = np.zeros(max_episode_num)

    path = os.path.join(experiment_path, folder_name)
    os.makedirs(path, exist_ok=True)
    model_path = os.path.join(path, 'model.pt')
    file_path = os.path.join(path, 'dic.npy')

    for episode in range(max_episode_num):
        fr_1 = envs.reset()
        fr_2 = envs.step(np.random.choice([2,3],num_envs))
        batch_input = preprocess_batch([fr_1, fr_2])
        #create lives vector, for turning into mask
        lives = np.array(num_envs*[n])

        rewards = np.zeros(num_envs, max_steps)
        episodes = np.ones(num_envs)*max_steps
        probs = np.zeros(num_envs, max_steps)

        if episode % 10000 == 0:
            np.save(file_path, data)
            if episode % 50000 == 0:
                torch.save(policy_net.state_dict(), model_path)


        for steps in range(max_steps):
            prob = policy_net.forward(batch_input, device)
            action = np.where(np.random.rand(num_envs) < prob, UP, DOWN)
            prob = np.where(action == UP, prob, 1.0 - prob)
            fr_1, r_1, _, _ = envs.step(action) #return rewards and new_state vector
            fr_2, r_2, _, _ = envs.step([0]*num_envs)
            reward = r_1 + r_2
            lost = np.where(reward < 0, 1, 0)
            episode_lost = np.where(lost == 1, steps, max_steps)
            episodes = np.minimum(episodes, episode_lost)
            lives -= lost
            reward = np.where(reward > 0, opp_rew, 0)

            if steps == max_steps-1:
                reward += win_reward
            rewards[:, steps] = reward
            probs[:, steps] = prob
            batch_input = preprocess_batch([fr_1,fr_2])

        reward_mask = np.where(lives > 0, 1, 0)
        rewards = rewards * reward_mask[:, None]

        data['rew'][episode] = np.mean(reward_mask)
        data['eps'][episode] = np.mean(episodes)
        policy_net.optimizer.zero_grad()
        return_gradient_entropy(rewards, probs, GAMMA, device)
        policy_net.optimizer.step()


    np.save(file_path, data)
    torch.save(policy_net.state_dict(), model_path)

"""import datetime
import os
import numpy as np

GAMMA = 1
episodes = 300000
max_steps = 100
lr = 1e-5

run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
results_path = os.path.join("scripts_workstation", "utils", "results")
os.makedirs(results_path, exist_ok = True)
experiment_path = os.path.join(results_path, "test")
os.makedirs(experiment_path, exist_ok = True)
print(os.getcwd())
run_path = os.path.join(experiment_path, run_timestamp)

rewards = train(GAMMA, episodes, max_steps, lr, run_path)"""
