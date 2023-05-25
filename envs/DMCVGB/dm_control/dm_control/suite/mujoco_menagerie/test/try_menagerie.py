import mujoco
from dm_control import suite
import matplotlib.pyplot as plt
import numpy as np
import imageio
from dm_control.suite.wrappers import action_scale, pixels

# model = mujoco.MjModel.from_xml_path("unitree_a1/a1.xml")
# data = mujoco.MjData(model)
# mujoco.mj_step(model, data)


def main():
    # model = mujoco.MjModel.from_xml_path("franka_emika_panda/panda.xml")
    # data = mujoco.MjData(model)
    # mujoco.mj_step(model, data)
    task_name = 'unitree'
    env = suite.load(task_name, 'walk')
    # print(timestep)
    random_state = np.random.RandomState(42)
    spec = env.action_spec()
    camera_id = 0
    render_kwargs = dict(height=224, width=224, camera_id=camera_id)
    env = pixels.Wrapper(env, pixels_only=True, render_kwargs=render_kwargs)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    timestep = env.reset()
    print(timestep.observation['pixels'].shape)

    # count = 0
    # while not timestep.last():
    #     action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
    #     timestep = env.step(action)
    #     count += 1
    # print(count)
    for j in range(1):
        images = []
        count = 0
        timestep = env.reset()
        while not timestep.last():
            action = random_state.uniform(env._action_spec.minimum, env._action_spec.maximum, spec.shape)
            timestep = env.step(action)
            images.append(timestep.observation['pixels'])
            print(timestep.reward)
            count += 1
        print(count)
        imageio.mimsave(
            f'./{task_name}_{j}.gif',
            [np.array(img) for i, img in enumerate(images) if i % 1 == 0], fps=15)

    # print(env.action_spec().sample())


if __name__ == '__main__':
    main()
