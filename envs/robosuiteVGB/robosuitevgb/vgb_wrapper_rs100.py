import robosuite as suite
from robosuite.controllers import load_controller_config
import gym
import numpy as np
from gym.spaces import Box
from typing import Union, List, Optional, Dict
import random
import copy
import xml.etree.ElementTree as ET
from .secant.envs.robosuite.custom_xml import CustomMujocoXML, XMLTextureModder
from .secant.utils.misc import render_img
from .secant.envs.robosuite.utils import get_obs_shape_from_dict
from .secant.envs.robosuite.preset_customization import (
    DEFAULT_TEXTURE_ALIAS,
    DEFAULT_TASK_TEXTURE_LIST,
    ALL_PRESET_ARGUMENTS,
    ALL_TEXTURE_PRESETS,
    ALL_COLOR_PRESETS,
    ALL_CAMERA_PRESETS,
    ALL_LIGHTING_PRESETS,
    get_custom_reset_config,
)
import numpy as np
from robosuite.utils.mjmod import CameraModder, LightingModder, TextureModder
from robosuite.wrappers import Wrapper
ALL_ROBOTS = list(suite.ALL_ROBOTS)


class VGBWrapper(gym.core.Env, Wrapper):
    """
    A gym style adapter for Robosuite
    """

    def __init__(
        self,
        env,
        color_randomization_args,
        camera_randomization_args,
        lighting_randomization_args,
        dynamics_randomization_args,
        task: str,
        obs_modality: Optional[List[str]] = ["rgb"],
        episode_length: int = 500,
        hard_reset: bool = False,
        channel_first: bool = True,
        camera_depths: bool = False,
        custom_reset_config: Optional[Union[Dict, str]] = None,
        mode: bool = "train",
        scene_id: Optional[int] = 0,
        verbose: bool = False,
        seed=None,
        randomize_color=True,  
        randomize_camera=True,
        randomize_lighting=True,
        randomize_dynamics=True,
        randomize_on_reset=True,
        moving_light=False,
    ):
        super().__init__(env)

        self.seed = seed
        if seed is not None:
            self.random_state = np.random.RandomState(seed)
        else:
            self.random_state = None
        self.randomize_color = randomize_color
        self.randomize_camera = randomize_camera
        self.randomize_lighting = randomize_lighting
        self.randomize_dynamics = randomize_dynamics
        self.color_randomization_args = color_randomization_args
        self.camera_randomization_args = camera_randomization_args
        self.lighting_randomization_args = lighting_randomization_args
        self.dynamics_randomization_args = dynamics_randomization_args
        self.randomize_on_reset = randomize_on_reset
        self.moving_light = moving_light

        self.modders = []

        # self._initialize_modders()

        # self.save_default_domain()

        if isinstance(scene_id, int):
            if verbose:
                print(f"{mode} scene_id: {scene_id}")
            custom_reset_config = get_custom_reset_config(
                task=task, mode=mode, scene_id=scene_id
            )
        self.task = task
        self.headless = True
        self.obs_modality = (
            ["rgb", "state"] if obs_modality is None else obs_modality.copy()
        )
        assert len(self.obs_modality) > 0, "Observation must have at least one modality"
        for modal in self.obs_modality:
            assert modal in [
                "rgb",
                "state",
            ], "Only 'rgb' and 'state' are supported as modality"
        if "rgb" in self.obs_modality:
            self._use_rgb = True
        else:
            self._use_rgb = False

        self._render_camera = "frontview"
        self._channel_first = channel_first
        self._use_depth = camera_depths
        self._max_episode_steps = episode_length
        self._hard_reset = hard_reset
        assert mode in ["train", "eval-easy", "eval-medium", "eval-hard"]
        self._mode = mode
        self._scene_id = scene_id

        self.env = env

        self.secant_modders = dict.fromkeys(["texture"])
        if custom_reset_config is not None:
            if isinstance(custom_reset_config, str):
                custom_reset_config = ALL_PRESET_ARGUMENTS.get(
                    custom_reset_config, None
                )
            self._reset_config = custom_reset_config
            xml_string = custom_reset_config.get("xml_string")
            self._initialize_xml_modder(custom_reset_config.get("custom_texture", None))

            if (
                xml_string is not None
                or custom_reset_config.get("custom_texture") is not None
            ):
                self.reset_xml_next = True
        else:
            self._reset_config = dict.fromkeys(
                [
                    "xml_string",
                    "custom_texture",
                ]
            )
            self.reset_xml_next = False

        obs_dict = self.env.observation_spec()
        self.observation_space = get_obs_shape_from_dict(self._reformat_obs(obs_dict))
        low, high = self.env.action_spec
        self.action_space = Box(low=low, high=high)

    def _initialize_modders(self):
        if self.randomize_color:
            self.tex_modder = TextureModder(
                sim=self.env.sim, random_state=self.random_state, **self.color_randomization_args
            )
            self.modders.append(self.tex_modder)

        if self.randomize_camera:
            self.camera_modder = CameraModder(
                sim=self.env.sim,
                random_state=self.random_state,
                **self.camera_randomization_args,
            )
            self.modders.append(self.camera_modder)

        if self.randomize_lighting:
            self.light_modder = LightingModder(
                sim=self.env.sim,
                random_state=self.random_state,
                **self.lighting_randomization_args,
            )
            self.modders.append(self.light_modder)

        # if self.randomize_dynamics:
        #     self.dynamics_modder = DynamicsModder(
        #         sim=self.env.sim,
        #         random_state=self.random_state,
        #         **self.dynamics_randomization_args,
        #     )
        #     self.modders.append(self.dynamics_modder)

    def _initialize_xml_modder(self, custom_texture: Optional[Union[str, dict]]):
        if custom_texture is not None:
            if isinstance(custom_texture, str):
                custom_texture = ALL_TEXTURE_PRESETS.get(custom_texture, None)
            config = custom_texture.copy()
            if custom_texture.get("tex_to_change", None) is None:
                config["tex_to_change"] = DEFAULT_TASK_TEXTURE_LIST[self.task]
            self.secant_modders["texture"] = XMLTextureModder(**config)


    def _reformat_obs(self, obs_dict, disable_channel_first=False):
        rgb_obs = {}
        state_obs = []
        reformatted = {}
        for name, obs in obs_dict.items():
            if name.endswith("_image") and "rgb" in self.obs_modality:
                view_name = name[:-6]
                if self._use_depth:
                    depth_obs = obs_dict[view_name + "_depth"]
                    obs = np.concatenate([obs, depth_obs[:, :, np.newaxis]], axis=2)
                obs = np.flipud(obs)
                if self._channel_first and not disable_channel_first:
                    obs = np.transpose(obs, (2, 0, 1))
                rgb_obs[view_name] = obs
            elif name.endswith("state") and "state" in self.obs_modality:
                state_obs.append(obs)
        if "rgb" in self.obs_modality:
            if len(rgb_obs.keys()) == 1:
                rgb_obs = list(rgb_obs.values())[0]
            reformatted["rgb"] = rgb_obs
        if "state" in self.obs_modality:
            reformatted["state"] = np.concatenate(state_obs)
        return reformatted

    def get_mujoco_xml(self):
        return CustomMujocoXML.build_from_env(self)

    def reset(
        self,
        xml_string: Optional[Union[str, List]] = None,
        custom_texture: Optional[Union[str, dict]] = None,
    ):

        if xml_string is not None:
            if xml_string != self._reset_config.get("xml_string", None):
                self._reset_config["xml_string"] = xml_string.copy()
            else:
                xml_string = None
        if custom_texture is not None:
            if custom_texture != self._reset_config.get("custom_texture", None):
                self._reset_config["custom_texture"] = copy.deepcopy(custom_texture)
                self._initialize_xml_modder(custom_texture)
            else:
                custom_texture = None
        # reset from xml should only be called if a different texture/xml is requested
        reset_from_xml = (
            xml_string is not None or custom_texture is not None or self.reset_xml_next
        )
        if reset_from_xml:
            self._reset_from_xml(xml_string)
            self.env.deterministic_reset = False
        else:
            self.env.reset()
        # self.env.reset()
        self.env._reset_internal()
        self.reset_xml_next = False

        if self.moving_light:
            self.step_pos = 0.4
            self.step_diffuse = 0.01
            self.light_pos_range = [self.env.sim.model.light_pos[:,1] - 20, self.env.sim.model.light_pos[:,1] + 20]
            self.light_diffuse_range = [max(self.env.sim.model.light_diffuse[:,1] - 0.5, 0.15), min(self.env.sim.model.light_diffuse[:,1] + 0.2, 0.95)]
        # TODO may change the order
        # self.restore_default_domain()
        # # save the original env parameters
        # self.save_default_domain()
        # reset counter for doing domain randomization at a particular frequency
        # self.step_counter = 0
        # update sims
        self._initialize_modders()
        for modder in self.modders:
            modder.update_sim(self.env.sim)
        self.randomize_domain()
        self.env.deterministic_reset = False
        self.env._reset_internal()
        return self._reformat_obs(self.env._get_observations(force_update=True))

    def _reset_from_xml(self, xml_string):
        if xml_string is not None:
            if isinstance(xml_string, list):
                xml_string = np.random.choice(xml_string)
            try:
                root = ET.fromstring(xml_string)
            except ET.ParseError:
                raise ValueError("Input xml_string is not a valid XML string")
            mujoco_xml = CustomMujocoXML.build_from_element(root)
        else:
            mujoco_xml = self.get_mujoco_xml()
        # f = open("twoarmsaaa.xml", 'w')
        # f.write(mujoco_xml.to_string())
        # f.close()

        if self.secant_modders["texture"] is not None:
            self.secant_modders["texture"].random_texture_change(mujoco_xml)  # TODO may change the texture
        self.env.reset_from_xml_string(mujoco_xml.to_string())

    def get_tex_candidate(self):
        """
        Get a tex_candidate dictionary from the current env. The tex_candidate
        dictionary can be passed to custom_texture in reset()
        """
        mujoco_xml = self.get_mujoco_xml()
        texture_list = DEFAULT_TASK_TEXTURE_LIST[self.task]
        tex_candidates = {}

        for alias in texture_list:
            mat_name = DEFAULT_TEXTURE_ALIAS[alias]
            texture = mujoco_xml.get_material_texture(mat_name=mat_name)
            tex_candidates[alias] = texture
        return tex_candidates

    def step(self, action):
        if self.moving_light:

            self.env.sim.model.light_diffuse[:, :] += self.step_diffuse
            self.env.sim.model.light_pos[:, :] += self.step_pos
            # yang_pos = self.env.sim.model.light_pos[:, :]
            # yang_diffuse = self.env.sim.model.light_diffuse[:, :]
            if self.env.sim.model.light_diffuse[:, 0] > self.light_diffuse_range[1] or self.env.sim.model.light_diffuse[:, 0] < self.light_diffuse_range[0]:
                self.step_diffuse *= -1
            if self.env.sim.model.light_pos[:, 0] > self.light_pos_range[1] or self.env.sim.model.light_pos[:, 0] < self.light_pos_range[0]:
                self.step_pos *= -1

        ob_dict, reward, done, info = self.env.step(action)
        info["mode"] = self._mode
        info["scene_id"] = self._scene_id
        return self._reformat_obs(ob_dict), reward, done, info

    def seed(self, seed=None):
        assert isinstance(seed, int), "seed must be an integer"
        np.random.seed(seed)
        random.seed(seed)

    def render(self, mode="human", render_camera=None, backend="cv2", waitkey=100):
        """
        Render the current frame. Note that due to robosuite's design, the specified
        backend (cv2/matplotlib) will be used to render "human" only if rgb is turned
        on for modality. Otherwise, robosuite's built-in renderer will be used.
        """
        if mode == "human":
            if self.headless:
                raise AssertionError("Can't render human in headless mode")
            elif self._use_rgb:
                img = self.render(mode="rgb_array")
                render_img(img, backend=backend, waitkey=waitkey)
            else:
                self.env.render()
        elif mode in ["rgb_array", "rgbd_array"]:
            assert self._use_rgb, "''rgb' must be included in obs_modality"
            rgb_obs = self._reformat_obs(
                self.env._get_observation(), disable_channel_first=True
            )["rgb"]
            if isinstance(rgb_obs, dict):
                render_camera = (
                    self._render_camera if render_camera is None else render_camera
                )
                assert (
                    render_camera in rgb_obs
                ), f"render_camera '{render_camera}' is not in observations"
                rgb_obs = rgb_obs[render_camera]
            if mode == "rgb_array":
                return rgb_obs[:, :, :3].astype(np.uint8)
            else:   # mode = rgbd_array
                assert self._use_depth, "Depth must be enabled at initialization"
                return rgb_obs
        else:
            raise AssertionError(
                f"mode should be one of: 'rgb_array', 'rgbd_array', "
                f"'human', received {mode}."
            )

    def close(self):
        self.env.close()


    def randomize_domain(self):
        """
        Runs domain randomization over the environment.
        """
        for modder in self.modders:
            modder.randomize()

    def save_default_domain(self):
        """
        Saves the current simulation model parameters so
        that they can be restored later.
        """
        for modder in self.modders:
            modder.save_defaults()

    def restore_default_domain(self):
        """
        Restores the simulation model parameters saved
        in the last call to @save_default_domain.
        """
        for modder in self.modders:
            modder.restore_defaults()
