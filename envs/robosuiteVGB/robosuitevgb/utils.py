from typing import Dict
from omegaconf import DictConfig
from hydra import compose, initialize
import robosuite as suite
from robosuitevgb.vgb_wrapper import VGBWrapper
from robosuite.controllers import load_controller_config
import hydra


def omegaconf_to_dict(d: DictConfig) -> Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret


def make_env(task_name: str, seed: int, scene_id: int = 0):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(config_path="../cfg"):
        cfg = compose(config_name="robo_config")
        cfg_dict = omegaconf_to_dict(cfg)
    assert scene_id >= 0 and scene_id <= 9
    cfg_dict['scene_id'] = scene_id
    cfg_dict['task_def']['controller_configs'] = load_controller_config(
        default_controller=cfg_dict['task_def']['controller_configs'])
    cfg_dict['task_def']['env_name'] = task_name
    if type(cfg_dict['task_def']['robots']) != str:
        robots = []
        for robot in cfg_dict['task_def']['robots']:
            robots.append(robot)
        cfg_dict['task_def']['robots'] = robots
    if "TwoArm" in cfg_dict['task_def']['env_name'] and type(cfg_dict['task_def']['robots']) == str:
        cfg_dict['task_def']['robots'] = [cfg_dict['task_def']['robots']] * 2
    env = suite.make(
        **cfg_dict['task_def'],
        use_camera_obs=True,
        has_offscreen_renderer=True,
    )
    if cfg_dict['mode'] == 'train':
        randomize_color = randomize_lighting = False
        randomize_dynamics = False
        randomize_camera = False
        cfg_dict['moving_light'] = False
        cfg_dict['except_robot'] = True
    else:
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    print('Now the mode is {}'.format(cfg_dict['mode']))
    env = VGBWrapper(env=env,
                     color_randomization_args=cfg_dict['color'],
                     camera_randomization_args=cfg_dict['camera'],
                     lighting_randomization_args=cfg_dict['light'],
                     dynamics_randomization_args=cfg_dict['dynamic'],
                     seed=seed,
                     task=cfg_dict['task_def']['env_name'],
                     obs_modality=cfg_dict['obs_modality'],
                     channel_first=cfg_dict['channel_first'],
                     camera_depths=cfg_dict['task_def']['camera_depths'],
                     episode_length=cfg_dict['task_def']['horizon'],
                     hard_reset=cfg_dict['task_def']['hard_reset'],
                     mode=cfg_dict['mode'],
                     scene_id=cfg_dict['scene_id'],
                     moving_light=cfg_dict['moving_light'],
                     randomize_color=randomize_color,
                     randomize_camera=randomize_camera,
                     randomize_lighting=randomize_lighting,
                     randomize_dynamics=randomize_dynamics,
                     video_background=cfg_dict['video_background'],
                     image_height=cfg_dict['image_height'],
                     image_width=cfg_dict['image_width'],
                     except_robot=cfg_dict['except_robot'],
                     )
    return env
