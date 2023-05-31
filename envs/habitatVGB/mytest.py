import gym
import numpy as np
from matplotlib import pyplot as plt
import habitat
from habitat import *
# import habitat.utils.gym_definitions
# import habitat.utils.make_env
from habitat_sim.utils import viz_utils as vut



def main():
    mode = 'train'
    seed = 1
    appearance_id = 'original'
    name = "HabitatImageNav-v0"

    env = make_env(name, mode=mode, seed=seed, appearance_id=appearance_id)
    obs = env.reset()

if __name__ == "__main__":
    main()
