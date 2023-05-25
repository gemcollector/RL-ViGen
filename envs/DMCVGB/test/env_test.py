from dm_control import suite
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import torch
import os
from dmcvgb.arguments import parse_args
from dmcvgb.make_env import make_env

import imageio
import torch
import argparse
from PIL import Image



def main(args):
    test_env = make_env()

    timestep = test_env.reset()
    timestep = test_env.reset()
    timestep = test_env.reset()
    timestep = test_env.reset()
    timestep = test_env.reset()
    timestep = test_env.reset()

    random_state = np.random.RandomState(42)
    # spec = test_env.action_spec()
    action_space = test_env.action_space
    images = []
    count = 0
    timestep = test_env.reset()
    obs = timestep.frames[1].swapaxes(0, 1).swapaxes(1, 2)
    Image.fromarray(obs).save('./unitree2.png')
    images.append(obs)
    plt.figure(dpi=300)
    plt.imshow(obs)
    plt.show()
    action = random_state.uniform(action_space.low, action_space.high, action_space.shape).astype(np.float32)
    timestep = test_env.step(action)


    while count < 200:
        # action = random_state.uniform(spec.minimum, spec.maximum, spec.shape).astype(np.float32)
        # action = random_state.uniform(action_space.low, action_space.high, action_space.shape).astype(np.float32)
        action = random_state.uniform(0, 0, action_space.shape).astype(np.float32)

        timestep = test_env.step(action)
        obs = timestep[0].frames[1].swapaxes(0, 1).swapaxes(1, 2)
        # timestep[0]: next_obs, timestep[1]: reward, timestep[2]: done, timestep[3]: info
        images.append(obs)
        count += 1
        # plt.figure(dpi=300)
        # plt.imshow(obs)
        # plt.show()
    imageio.mimsave('h_comp.gif', [np.array(img) for i, img in enumerate(images) if i % 1 == 0], fps=15)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    # suite.load('unitree', 'walk')
    # test dmc (added franka and unitree)
    # random_state = np.random.RandomState(42)
    # # env = suite.load('franka', 'push', task_kwargs={'random': random_state})
    # env = suite.load('franka', 'reach', task_kwargs={'random': random_state})
    # action_spec = env.action_spec()
    # timestep = env.reset()
    # # obs = timestep.observation
    # # images = []
    # for _ in range(1):
    #     action = random_state.uniform(action_spec.minimum, action_spec.maximum, action_spec.shape)
    #     timestep = env.step(action)
    #     # obs = timestep.observation
    #     camera0 = env.physics.render(camera_id=0, height=200, width=200)
    #     plt.figure(dpi=300)
    #     plt.imshow(camera0)
    #     plt.show()
    #     if timestep.last():
    #         # print(timestep.last())
    #         timestep = env.reset()




