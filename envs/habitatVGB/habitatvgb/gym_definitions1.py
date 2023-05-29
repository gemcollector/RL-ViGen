#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from glob import glob
from typing import TYPE_CHECKING, Any, List, Optional
import os
import shutil
import numpy as np
import gym
from gym.envs.registration import register, registry
import json
import habitat
import habitat.utils.env_utils
from habitat.config.default import _HABITAT_CFG_DIR
from habitat.config.default_structured_configs import ThirdRGBSensorConfig
from habitat.core.environments import get_env_class
from habitat_sim.gfx import LightInfo, LightPositionModel
from typing import Dict
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize


def omegaconf_to_dict(d: DictConfig) -> Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret
if TYPE_CHECKING:
    from omegaconf import DictConfig


gym_task_config_dir = osp.join(_HABITAT_CFG_DIR, "benchmark/")


def _get_gym_name(cfg: "DictConfig") -> Optional[str]:
    if "habitat" in cfg:
        cfg = cfg.habitat
    if "gym" in cfg and "auto_name" in cfg["gym"]:
        return cfg["gym"]["auto_name"]
    return None


def _get_env_name(cfg: "DictConfig") -> Optional[str]:
    if "habitat" in cfg:
        cfg = cfg.habitat
    return cfg["env_task"]


def make_gym_from_config(config: "DictConfig") -> gym.Env:
    """
    From a habitat-lab or habitat-baseline config, create the associated gym environment.
    """
    if "habitat" in config:
        config = config.habitat
    env_class_name = _get_env_name(config)
    env_class = get_env_class(env_class_name)
    assert (
        env_class is not None
    ), f"No environment class with name `{env_class_name}` was found, you need to specify a valid one with env_task"
    return habitat.utils.env_utils.make_env_fn(
        env_class=env_class, config=config
    )


def _make_habitat_gym_env(
    cfg_file_path: str,
    override_options: List[Any] = None,
    use_render_mode: bool = False,
    mode: str = 'train',
) -> gym.Env:
    if override_options is None:
        override_options = []

    config = habitat.get_config(cfg_file_path, overrides=override_options)
    OmegaConf.set_readonly(config, False)

    with initialize(config_path="../cfg"):
        cfg = compose(config_name="config")
        cfg_dict = omegaconf_to_dict(cfg)
    random_state = np.random.RandomState(cfg_dict['seed'])
    # if cfg_dict['appearance_id'] != 'original':
    src_file = os.path.join(f'data/scene_datasets/habitat-test-scenes/{cfg_dict["appearance_id"]}', 'skokloster-castle.glb')
    src_dir = os.path.dirname(src_file)
    dst_dir = os.path.dirname(src_dir)
    dst_file = os.path.join(dst_dir, os.path.basename(src_file))
    shutil.copy(src_file, dst_file)

    cfg_dict['mode'] = mode
    if mode == 'test':
        config.habitat.dataset.data_path = cfg_dict['setting']['data_path'][cfg_dict['mode']][cfg_dict["scene_id"]]
    else:
        config.habitat.dataset.data_path = cfg_dict['setting']['data_path'][cfg_dict['mode']]
    # if mode == 'test':
    #     import pdb; pdb.set_trace()
    #     config.habitat.dataset.data_path = config.habitat.dataset.data_path.replace('id', f'{cfg_dict["scene_id"]}')  # TODO change json to the corrsponding dir
    config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = cfg_dict['image_height']
    config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = cfg_dict['image_width']

    with open('data/default.physics_config.json', 'r') as f:
        dynamics = json.load(f)
    dynamics['gravity'][0] = random_state.uniform(
        cfg_dict['setting']['dynamics']['gravity'][cfg_dict['dynamics']['gravity']]['x'][0],
        cfg_dict['setting']['dynamics']['gravity'][cfg_dict['dynamics']['gravity']]['x'][1])
    dynamics['gravity'][1] = random_state.uniform(
        cfg_dict['setting']['dynamics']['gravity'][cfg_dict['dynamics']['gravity']]['y'][0],
        cfg_dict['setting']['dynamics']['gravity'][cfg_dict['dynamics']['gravity']]['y'][1])
    dynamics['gravity'][2] = random_state.uniform(
        cfg_dict['setting']['dynamics']['gravity'][cfg_dict['dynamics']['gravity']]['z'][0],
        cfg_dict['setting']['dynamics']['gravity'][cfg_dict['dynamics']['gravity']]['z'][1])
    dynamics['friction_coefficient'] = random_state.uniform(
        cfg_dict['setting']['dynamics']['friction_coefficient'][cfg_dict['dynamics']['friction_coefficient']][0],
        cfg_dict['setting']['dynamics']['friction_coefficient'][cfg_dict['dynamics']['friction_coefficient']][1])
    dynamics['restitution_coefficient'] = random_state.uniform(
        cfg_dict['setting']['dynamics']['restitution_coefficient'][cfg_dict['dynamics']['restitution_coefficient']][0],
        cfg_dict['setting']['dynamics']['restitution_coefficient'][cfg_dict['dynamics']['restitution_coefficient']][1])
    with open('data/default.physics_config.json', 'w') as f:
        json.dump(dynamics, f)

    config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position[0] = random_state.uniform(
        cfg_dict['setting']['camera']['position'][cfg_dict['camera']['position']]['x'][0],
        cfg_dict['setting']['camera']['position'][cfg_dict['camera']['position']]['x'][1])
    config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position[1] = random_state.uniform(
        cfg_dict['setting']['camera']['position'][cfg_dict['camera']['position']]['y'][0],
        cfg_dict['setting']['camera']['position'][cfg_dict['camera']['position']]['y'][1])
    config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position[2] = random_state.uniform(
        cfg_dict['setting']['camera']['position'][cfg_dict['camera']['position']]['z'][0],
        cfg_dict['setting']['camera']['position'][cfg_dict['camera']['position']]['z'][1])

    config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.orientation[0] = random_state.uniform(
        cfg_dict['setting']['camera']['orientation'][cfg_dict['camera']['orientation']][0],
        cfg_dict['setting']['camera']['orientation'][cfg_dict['camera']['orientation']][1])
    config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.orientation[1] = random_state.uniform(
        cfg_dict['setting']['camera']['orientation'][cfg_dict['camera']['orientation']][0],
        cfg_dict['setting']['camera']['orientation'][cfg_dict['camera']['orientation']][1])
    config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.orientation[2] = random_state.uniform(
        cfg_dict['setting']['camera']['orientation'][cfg_dict['camera']['orientation']][0],
        cfg_dict['setting']['camera']['orientation'][cfg_dict['camera']['orientation']][1])

    config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = random_state.randint(
        cfg_dict['setting']['camera']['hfov'][cfg_dict['camera']['hfov']][0],
        cfg_dict['setting']['camera']['hfov'][cfg_dict['camera']['hfov']][1])


    if use_render_mode:
        with habitat.config.read_write(config):
            sim_config = config.habitat.simulator
            default_agent_name = sim_config.agents_order[
                sim_config.default_agent_id
            ]
            default_agent = sim_config.agents[default_agent_name]
            if len(sim_config.agents) == 1:
                default_agent.sim_sensors.update(
                    {"third_rgb_sensor": ThirdRGBSensorConfig()}
                )
            else:
                default_agent.sim_sensors.update(
                    {
                        "default_agent_third_rgb_sensor": ThirdRGBSensorConfig(
                            uuid="default_robot_third_rgb"
                        )
                    }
                )
    env = make_gym_from_config(config)
    # if cfg_dict['scene_id'] == 0:
    self = env.env._env._env
    self._sim.new_sim_config = self._sim.create_sim_config(self._sim._sensor_suite)
    pos_x = random_state.uniform(
        cfg_dict['setting']['light']['position'][cfg_dict['light']['position']]['x'][0],
        cfg_dict['setting']['light']['position'][cfg_dict['light']['position']]['x'][1])
    pos_y = random_state.uniform(
        cfg_dict['setting']['light']['position'][cfg_dict['light']['position']]['y'][0],
        cfg_dict['setting']['light']['position'][cfg_dict['light']['position']]['y'][1])
    pos_z = random_state.uniform(
        cfg_dict['setting']['light']['position'][cfg_dict['light']['position']]['z'][0],
        cfg_dict['setting']['light']['position'][cfg_dict['light']['position']]['z'][1])
    intensity = random_state.uniform(
        cfg_dict['setting']['light']['intensity'][cfg_dict['light']['intensity']][0],
        cfg_dict['setting']['light']['intensity'][cfg_dict['light']['intensity']][1])
    color_r = random_state.uniform(cfg_dict['setting']['light']['color'][cfg_dict['light']['color']][0],
                                   cfg_dict['setting']['light']['color'][cfg_dict['light']['color']][
                                       1]) * intensity
    color_g = random_state.uniform(cfg_dict['setting']['light']['color'][cfg_dict['light']['color']][0],
                                   cfg_dict['setting']['light']['color'][cfg_dict['light']['color']][
                                       1]) * intensity
    color_b = random_state.uniform(cfg_dict['setting']['light']['color'][cfg_dict['light']['color']][0],
                                   cfg_dict['setting']['light']['color'][cfg_dict['light']['color']][
                                       1]) * intensity
    # import pdb;pdb.set_trace()
    my_scene_lighting_setup = [
        LightInfo(vector=[pos_x, pos_y, pos_z, 0.0], color=[color_r, color_g, color_b], model=LightPositionModel.Camera)
    ]
    # import habitat_sim
    # self._sim.new_sim_config.sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
    # import pdb;pdb.set_trace()
    self._sim.set_light_setup(my_scene_lighting_setup, "vgb_scene_lighting")
    self._sim.new_sim_config.sim_cfg.scene_light_setup = "vgb_scene_lighting"
    self._sim.new_sim_config.sim_cfg.override_scene_light_defaults = True
    self._sim.super_reconfigure(self._sim.new_sim_config)
    # obj_template_mgr = self._sim.get_object_template_manager()
    # # get the rigid object manager, which provides direct
    # # access to objects
    # rigid_obj_mgr = self._sim.get_rigid_object_manager()
    # data_path = 'da'
    # # load some object templates from configuration files
    # sphere_template_id = obj_template_mgr.load_configs(
    #     "data/test_assets/objects/sphere"
    # )[0]
    # chair_template_id = obj_template_mgr.load_configs(
    #     "data/test_assets/objects/chair"
    # )[0]
    #
    # # create a sphere and place it at a desired location
    # obj_1 = rigid_obj_mgr.add_object_by_template_id(sphere_template_id)
    # obj_1.translation = [-6.544811725616455, 0.00991860032081604, 24.039745330810547]
    return env


def _try_register(id_name, entry_point, kwargs):
    if id_name in registry.env_specs:
        return
    register(
        id_name,
        entry_point=entry_point,
        kwargs=kwargs,
    )


if "Habitat-v0" not in registry.env_specs:
    # Generic supporting general configs
    _try_register(
        id_name="Habitat-v0",
        entry_point="habitatvgb.gym_definitions:_make_habitat_gym_env",
        kwargs={},
    )

    _try_register(
        id_name="HabitatRender-v0",
        entry_point="habitatvgb.gym_definitions:_make_habitat_gym_env",
        kwargs={"use_render_mode": True},
    )

    hab_baselines_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    gym_template_handle = "Habitat%s-v0"
    render_gym_template_handle = "HabitatRender%s-v0"

    for fname in glob(
        osp.join(gym_task_config_dir, "**/*.yaml"), recursive=True
    ):
        full_path = osp.join(gym_task_config_dir, fname)
        if not fname.endswith(".yaml"):
            continue
        cfg_data = habitat.get_config(full_path)
        gym_name = _get_gym_name(cfg_data)
        if gym_name is not None:
            # Register this environment name with this config
            _try_register(
                id_name=gym_template_handle % gym_name,
                entry_point="habitatvgb.gym_definitions:_make_habitat_gym_env",
                kwargs={"cfg_file_path": full_path},
            )

            _try_register(
                id_name=render_gym_template_handle % gym_name,
                entry_point="habitatvgb.gym_definitions:_make_habitat_gym_env",
                kwargs={"cfg_file_path": full_path, "use_render_mode": True},
            )
