import gym
from collections import deque
from typing import Any, NamedTuple
from dm_env import StepType, specs
import collections
import dm_env
from carlaenv.utils import make_env_10, make_env_10_eval
import numpy as np


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)

class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
    def reset(self):
        obs = self._env.reset()
        time_step = collections.OrderedDict()
        time_step['observation'] = obs
        time_step['action'] = np.array([0. for _ in range(2)])
        time_step['step_type'] = StepType.MID
        time_step['reward'] = 0.0
        time_step['discount'] = 1.0
        return self._augment_time_step(time_step)

    def step(self, action):
        # time_step = self._env.step(action)
        next_obs, reward, done, info = self._env.step(action)
        time_step = collections.OrderedDict()
        time_step['observation'] = next_obs
        time_step['action'] = np.float32(action)
        time_step['reward'] = reward
        time_step['discount'] = 1.0
        if done == True:
            time_step['step_type'] = StepType.LAST
        else:
            time_step['step_type'] = StepType.MID
        return self._augment_time_step(time_step, np.float32(action)), info

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        time_step['action'] = np.float32(time_step['action'])
        return ExtendedTimeStep(observation=time_step['observation'],
                                step_type=time_step['step_type'],
                                action=time_step['action'],
                                reward=time_step['reward'] or 0.0,
                                discount=time_step['discount'] or 1.0)

    def observation_spec(self):
        return specs.BoundedArray(shape=(9, 84, 84), dtype=np.uint8, name='observation', minimum=0, maximum=255)

    def action_spec(self):
        return specs.BoundedArray(shape=(2, ), dtype=np.float32, name='action', minimum=-1.0, maximum=1.0)

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    
def carla_make(action_repeat):
    env = make_env_10(action_repeat)
    env = ExtendedTimeStepWrapper(FrameStack(env, 3))
    return env
    
    
def carla_make_eval(action_repeat):
    env = make_env_10_eval(action_repeat)
    env = ExtendedTimeStepWrapper(FrameStack(env, 3))
    return env
    
    
