import gym
import numpy as np
from typing import Optional


__all__ = ["TimeLimit", "AddHorizon"]


class TimeLimit(gym.Wrapper):
    """
    modified from gym.wrappers.TimeLimit
    """

    def __init__(self, env, max_episode_steps: Optional[int] = None):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    @property
    def max_episode_steps(self) -> Optional[int]:
        return self._max_episode_steps

    @property
    def elapsed_steps(self) -> Optional[int]:
        return self._elapsed_steps

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if (
            self._max_episode_steps is not None
            and self._elapsed_steps >= self._max_episode_steps
        ):
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class AddHorizon(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.max_episode_steps = env.max_episode_steps
        self._steps = 0
        assert isinstance(self.observation_space, gym.spaces.Dict)
        self.observation_space.spaces["horizon"] = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self._steps = 0
        obs["horizon"] = self._make_horizon(1.0)
        return obs

    def step(self, action):
        obs, *others = super().step(action)
        self._steps += 1
        h = (self.max_episode_steps - self._steps) / self.max_episode_steps
        obs["horizon"] = self._make_horizon(h)
        return (obs, *others)

    def _make_horizon(self, h):
        assert h >= 0
        return np.array([h], dtype=np.float32)
