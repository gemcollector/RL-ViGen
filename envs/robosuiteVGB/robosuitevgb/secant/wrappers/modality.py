import collections
import gym
import numpy as np


class SingleModality(gym.ObservationWrapper):
    def __init__(self, env, modality: str):
        super().__init__(env)
        self._modality = modality
        if hasattr(self.env, "observation_space") and isinstance(
            self.env.observation_space, gym.spaces.Dict
        ):
            self.observation_space = self.env.observation_space[modality]

    def observation(self, obs):
        assert isinstance(obs, collections.abc.Mapping)
        return obs[self._modality]


class RGBOnly(SingleModality):
    def __init__(self, env):
        super().__init__(env, modality="rgb")


class RGBFloat2Int(gym.ObservationWrapper):
    def observation(self, obs):
        obs = obs.copy()  # shallow copy
        if "rgb" in obs:
            obs["rgb"] = (obs["rgb"] * 255.0).astype(np.uint8)
        return obs
