from robosuitevgb import *
from PIL import Image
import imageio
import numpy as np


def main():
    env = make_env(task_name='Door', seed=1)
    for num in range(2):
        count = 0
        images = []
        done = False
        obs = env.reset()
        obs = obs['rgb']
        
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            obs = obs['rgb']
            img = obs.transpose(1,2,0)[:,:,:3]
            # Image.fromarray(img).save('./robosuitevgb1.png')
            images.append(img)
            count += 1

if __name__ == '__main__':
    main()