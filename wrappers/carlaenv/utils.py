from typing import Dict
from omegaconf import DictConfig
# from carlaenv.carla_env import CarlaEnv
from carlaenv.carla_env_10 import CarlaEnv10
from carlaenv.carla_env_10_eval import CarlaEnv10_eval
from hydra import compose, initialize
import hydra

def make_env():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(config_path="../../cfgs"):
        # cfg = compose(config_name="config", overrides=[f"task={task}"])
        cfg = compose(config_name="carlaenv_config")
        cfg_dict = omegaconf_to_dict(cfg)
    env = CarlaEnv(cfg_dict)
    return env

def make_env_8():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(config_path="../../cfgs"):
        # cfg = compose(config_name="config", overrides=[f"task={task}"])
        cfg = compose(config_name="carlaenv8_config")
        cfg_dict = omegaconf_to_dict(cfg)
    env = CarlaEnv8(cfg_dict)
    return env

def make_env_10(action_repeat):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(config_path="../../cfgs"):
        # cfg = compose(config_name="config", overrides=[f"task={task}"])
        cfg = compose(config_name="carlaenv10_config")
        cfg_dict = omegaconf_to_dict(cfg)
    cfg_dict['frame_skip'] = action_repeat
    env = CarlaEnv10(cfg_dict)
    return env


def make_env_10_eval(action_repeat):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(config_path="../../cfgs"):
        # cfg = compose(config_name="config", overrides=[f"task={task}"])
        cfg = compose(config_name="carlaenv10_eval_config")
        cfg_dict = omegaconf_to_dict(cfg)
    cfg_dict['frame_skip'] = action_repeat
    if cfg_dict['scenario'] == 'tunnel':
        cfg_dict['num_other_cars'] = 5
        cfg_dict['num_other_cars_nearby'] = 2
    elif cfg_dict['scenario'] == 'narrow' or cfg_dict['scenario'] == 'roundabout':
        cfg_dict['num_other_cars'] = 5
        cfg_dict['num_other_cars_nearby'] = 0
    env = CarlaEnv10_eval(cfg_dict)
    return env


def omegaconf_to_dict(d: DictConfig) -> Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret