from robosuitevgb import *
from PIL import Image
import imageio
import numpy as np


def main():
    env = make_env()
    for aaa in range(2):
        count = 0
        images = []
        done = False
        obs = env.reset()
        obs = obs['rgb']
        img = obs.transpose(1, 2, 0)[:, :, :3]
        # yang = img.astype(np.uint8)
        # Image.fromarray(img).save('./robosuitevgb.png')  # .astype(np.uint8)
        images.append(img)
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            obs = obs['rgb']
            img = obs.transpose(1,2,0)[:,:,:3]
            Image.fromarray(img).save('./robosuitevgb1.png')
            images.append(img)
            count += 1
        print(count)
        imageio.mimsave(f'{aaa}.gif', [np.array(img) for i, img in enumerate(images) if i % 1 == 0], fps=15)

if __name__ == '__main__':
    # t0 = time.time()
    main()
    # t1 = time.time()
    # print(t1 - t0)