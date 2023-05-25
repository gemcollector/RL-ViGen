import gym
import gym.spaces
import collections
import tree

from typing import Optional, List

import numpy as np
from typing_extensions import Literal
from .utils import map_gym_space


__all__ = ["FrameStack"]


class FrameStack(gym.Wrapper):
    def __init__(
        self,
        env,
        k: int,
        mode: Literal["stack", "concat"] = "concat",
        stack_dim: int = 0,
        include_keys: Optional[List[str]] = None,
    ):
        """
        Args:
            mode: 'stack' creates a new dim, while 'concat' concatenates the leading dim
                e.g. stack([7,9]) twice -> [2, 7, 9]
                     concat([7,9]) twice -> [14, 9]
            stack_dim: which axis to stack
            include_keys: frame stack only the included keys, otherwise framestack all
        """
        super().__init__(env)
        self._k = k
        self._frames = collections.deque([], maxlen=k)
        assert mode in ["stack", "concat"]
        self._mode = mode
        self._stack_dim = stack_dim
        self.observation_space = map_gym_space(
            self._transform_space, env.observation_space
        )
        self._include_keys = include_keys
        if include_keys:
            if isinstance(include_keys, str):
                self._include_keys = [include_keys]

        if hasattr(env, "_max_episode_steps"):
            self._max_episode_steps = env._max_episode_steps

    def _compute_obs_shape(self, shape):
        assert len(shape) >= 1
        shape = tuple(shape)
        d = self._stack_dim
        k = self._k
        if self._mode == "stack":
            assert (
                self._stack_dim < len(shape) + 1
            ), f"stack_dim {d} must <= the shape dim"
            return shape[:d] + (k,) + shape[d:]
        elif self._mode == "concat":
            assert self._stack_dim < len(shape), (
                f"stack_dim {d} must be " f"smaller than the shape dim"
            )
            return shape[:d] + (k * shape[d],) + shape[d + 1 :]
        else:
            raise NotImplementedError(f"Unknown frame stack mode: {self._mode}")

    def _transform_space(self, space):
        if isinstance(space, gym.spaces.Box):
            low, high = space.low, space.high
            _stack = np.stack if self._mode == "stack" else np.concatenate
            low = _stack([low] * self._k, axis=self._stack_dim)
            high = _stack([high] * self._k, axis=self._stack_dim)

            return gym.spaces.Box(
                low=low,
                high=high,
                shape=self._compute_obs_shape(space.shape),
                dtype=space.dtype,
            )
        else:
            raise NotImplementedError(
                f"Unsupported space: {space}. "
                f"FrameStack only supports Box space and recursive structures of Box"
            )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        frames = list(self._frames)
        _stack = any_stack if self._mode == "stack" else any_concat
        if self._include_keys:
            obs_to_stack = [
                {k: frame[k] for k in self._include_keys} for frame in self._frames
            ]
            obs = _stack(obs_to_stack, dim=self._stack_dim)
            # for non-stacking keys, include the values from the last frame
            for k, v in frames[-1].items():
                if k not in self._include_keys:
                    obs[k] = v
            return obs
        else:
            return _stack(frames, dim=self._stack_dim)


def any_stack(xs: List, *, dim: int = 0):
    """
    Works for both torch Tensor and numpy array
    """

    def _any_stack_helper(*xs):
        x = xs[0]
        if isinstance(x, np.ndarray):
            return np.stack(xs, axis=dim)
        elif torch.is_tensor(x):
            return torch.stack(xs, dim=dim)
        elif isinstance(x, float):
            # special treatment for float, defaults to float32
            return np.array(xs, dtype=np.float32)
        else:
            return np.array(xs)

    return tree.map_structure(_any_stack_helper, *xs)


def any_concat(xs: List, *, dim: int = 0):
    """
    Works for both torch Tensor and numpy array
    """

    def _any_concat_helper(*xs):
        x = xs[0]
        if isinstance(x, np.ndarray):
            return np.concatenate(xs, axis=dim)
        elif torch.is_tensor(x):
            return torch.cat(xs, dim=dim)
        elif isinstance(x, float):
            # special treatment for float, defaults to float32
            return np.array(xs, dtype=np.float32)
        else:
            return np.array(xs)

    return tree.map_structure(_any_concat_helper, *xs)
