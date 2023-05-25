import gym
import habitatvgb.gym_definitions
import os
import hydra


def make_env(task, mode, seed, appearance_id='original'):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    origin_dir = os.getcwd()
    current_dir = os.path.dirname(__file__)
    os.chdir(current_dir + '/../../..')
    env = gym.make(task, mode=mode, seed=seed, appearance_id=appearance_id)
    os.chdir(origin_dir)
    return env
