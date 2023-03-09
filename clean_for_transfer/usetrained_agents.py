import torch
import numpy as np
import gym

from pong_utils import preprocess_batch
from pong_utils import preprocess_single
from pong_utils import Policy
import os
import random as rand

RIGHT = 4
LEFT = 5

folder = 'eps1000n8curricgeneral'

path = os.path.join('folder', 'model.pt')
init_path = os.path.join('pong_curriculum', '202303-0814-4634')
final_path = os.path.join('results', init_path)

print(torch.cuda.device_count())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(f'is gpu available: {torch.cuda.is_available()}')
print(f'device count: {torch.cuda.device_count()}')

def preprocess_batch(images, bkg_color=np.array([144, 72, 17])):
    list_of_images = np.asarray(images)
    if len(list_of_images.shape) < 5:
        list_of_images = np.expand_dims(list_of_images, 1)
    # subtract bkg and crop
    list_of_images_prepro = np.mean(list_of_images[:, :, 34:-16:2, ::2] - bkg_color,
                                    axis=-1) / 255.
    batch_input = np.swapaxes(list_of_images_prepro, 0, 1)
    return batch_input

env = gym.make("PongDeterministic-v4", seed = 1234)
#first = env.reset()

policy = Policy.to(device)
policy.load_state_dict(torch.load(final_path, map_location=device))

def play(env, policy, time=2000, nrand=5):
    env.reset()

    # star game
    env.step(1)

    # perform nrand random steps in the beginning
    for _ in range(nrand):
        frame1, reward1, is_done, _ = env.step(np.random.choice([RIGHT, LEFT]))
        frame2, reward2, is_done, _ = env.step(0)

    anim_frames = []

    for t in range(time):

        env.render()
        frame_input = preprocess_batch([frame1, frame2])
        prob = policy(frame_input)

        # RIGHT = 4, LEFT = 5
        action = RIGHT if rand.random() < prob else LEFT
        frame1, r1, is_done, _ = env.step(action)
        frame2, r2, is_done, _ = env.step(0)
        reward = r1 + r2

        if reward < 0:
            print((t, 'loss'))
        if reward > 0:
            print((t, 'win'))

        if is_done:
            print((t, 'done'))
            break

    env.close()

    return


