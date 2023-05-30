import time
import numpy as np
import imageio
# from dhm import *
import dhm
import matplotlib.pyplot as plt
from PIL import Image

def main():
    train_env = dhm.make_env_RRL('hammer-v0')

    timestep = train_env.reset()
    plt.figure(dpi=300)
    plt.imshow(timestep.observation.transpose(1, 2, 0)[:, :, :3])
    plt.show()
    # Image.fromarray(timestep.observation.transpose(1, 2, 0)[:, :, :3]).save('./pen5.png')
    random_state = np.random.RandomState(42)
    spec = train_env.action_spec()
    images = []
    count = 0
    while not timestep.last() and count < 400:
        action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
        timestep = train_env.step(action)
        # train_env.render()
        # obs = timestep.observation
        # plt.figure(dpi=300)
        # plt.imshow(timestep.observation.transpose(1, 2, 0)[:, :, :])
        # plt.show()
        images.append(timestep.observation.transpose(1, 2, 0)[:, :, :3])
        count += 1
    print(count)
    imageio.mimsave('h_comp.gif', [np.array(img) for i, img in enumerate(images) if i % 1 == 0], fps=15)



if __name__ == '__main__':
    # t0 = time.time()
    main()
    # t1 = time.time()
    # print(t1 - t0)