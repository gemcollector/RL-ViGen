from dm_env import specs, Environment, TimeStep, StepType
from dmc import ActionDTypeWrapper, ActionRepeatWrapper, ExtendedTimeStepWrapper
from dm_control.suite.wrappers import action_scale
from collections import deque
from dm_env import StepType, specs
import dm_env
import numpy as np
from dmcvgb.make_env import make_env

class Gym2DMC(Environment):
    def __init__(self, gym_env) -> None:
        gym_obs_space = gym_env.observation_space
        # gym_obs_space = gym_env.observation_spec()[render_camera]
        self._observation_spec = specs.BoundedArray(
            shape=gym_obs_space.shape,
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation'
        )
        self._action_spec = specs.BoundedArray(
            shape=gym_env.action_space.shape,
            dtype=np.float32,
            minimum=gym_env.action_space.low,
            maximum=gym_env.action_space.high,
            name='action'
        )
        self._gym_env = gym_env

    def render(self):
        self._gym_env.render()

    def step(self, action):
        obs, reward, done, info = self._gym_env.step(action)
        obs = np.array(obs)

        if done:
            step_type = StepType.LAST
            discount = 0.0
        else:
            step_type = StepType.MID
            discount = 1.0

        return TimeStep(step_type=step_type,
                        reward=reward,
                        discount=discount,
                        observation=obs)

    def reset(self):
        obs = self._gym_env.reset()
        obs = np.array(obs)
        return TimeStep(step_type=StepType.FIRST,
                        reward=None,
                        discount=None,
                        observation=obs)

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)

        wrapped_obs_spec = env.observation_spec()

        pixels_shape = wrapped_obs_spec.shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[0] * num_frames], pixels_shape[1:]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(name, frame_stack=3, action_repeat=2, seed=1, type='original', difficulty='easy'):
    # create environment instance
    domain, task = name.split('_', 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup='ball_in_cup').get(domain, domain)
    env = make_env(domain_name=domain, task_name=task, seed=seed, action_repeat=action_repeat, frame_stack=frame_stack, type=type, difficulty=difficulty)
    # add wrappers
    env = Gym2DMC(env)
    env = ActionDTypeWrapper(env, np.float32)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # add renderings for clasical tasks
    # stack several frames
    # env = FrameStackWrapper(env, frame_stack)
    env = ExtendedTimeStepWrapper(env)
    return env







