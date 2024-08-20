import robosuite as suite
from robosuite.utils.mjmod import LightingModder, CameraModder, TextureModder
from robosuite.controllers import load_controller_config
import gym
import numpy as np
from gym.spaces import Box
from typing import Union, List, Optional, Dict
import random
import copy
import xml.etree.ElementTree as ET
from .custom_xml import CustomMujocoXML, XMLTextureModder
# from secant.utils.misc import render_img
from .utils import get_obs_shape_from_dict
from .preset_customization import (
    DEFAULT_TEXTURE_ALIAS,
    DEFAULT_TASK_TEXTURE_LIST,
    ALL_PRESET_ARGUMENTS,
    ALL_TEXTURE_PRESETS,
    ALL_COLOR_PRESETS,
    ALL_CAMERA_PRESETS,
    ALL_LIGHTING_PRESETS,
    get_custom_reset_config,
)

ALL_ROBOTS = list(suite.ALL_ROBOTS)


class RobosuiteAdapter(gym.core.Env):
    """
    A gym style adapter for Robosuite
    """

    def __init__(
        self,
        task: str,
        robots: Union[str, List[str]] = "Panda",
        controller_configs: Optional[Union[Dict, List[Dict]]] = None,
        controller_types: Optional[Union[str, List[str]]] = "OSC_POSE",
        headless: bool = True,
        obs_modality: Optional[List[str]] = ["rgb"],
        render_camera: str = "frontview",
        control_freq: int = 20,
        episode_length: int = 500,
        ignore_done: bool = True,
        hard_reset: bool = False,
        obs_cameras: Union[str, List[str]] = "agentview",
        channel_first: bool = True,
        image_height: int = 168,
        image_width: int = 168,
        camera_depths: bool = False,
        custom_reset_config: Optional[Union[Dict, str]] = None,
        mode: bool = "train",
        scene_id: Optional[int] = 0,
        reward_shaping: bool = True,
        verbose: bool = False,
        **kwargs,
    ):

        if controller_types:
            if isinstance(controller_types, str):
                controller_types = [controller_types]
            if "Two" in task and len(controller_types) == 1:
                controller_types = controller_types * 2
            controller_configs = []
            for controller_type in controller_types:
                assert controller_type in ["OSC_POSE", "JOINT_VELOCITY"]
                controller_configs.append(
                    load_controller_config(default_controller=controller_type)
                )

        if isinstance(robots, str):
            robots = [robots]
        if "Two" in task and len(robots) == 1:
            robots = robots * 2
        for robot in robots:
            assert robot in ALL_ROBOTS, f"Robot {robot} does not exist."

        if isinstance(scene_id, int):
            if verbose:
                print(f"{mode} scene_id: {scene_id}")
            custom_reset_config = get_custom_reset_config(
                task=task, mode=mode, scene_id=scene_id
            )

        self.task = task
        self.robots = robots
        self.headless = headless
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
            use_camera_obs = True
            has_offscreen_renderer = True
            has_renderer = False
            self._use_rgb = True
        else:
            use_camera_obs = False
            has_offscreen_renderer = False
            has_renderer = not self.headless
            self._use_rgb = False

        self._render_camera = render_camera
        self._channel_first = channel_first
        self._use_depth = camera_depths
        self._max_episode_steps = episode_length
        self._hard_reset = hard_reset
        assert mode in ["train", "eval-easy", "eval-medium", "eval-hard"]
        self._mode = mode
        self._scene_id = scene_id

        self.env = suite.make(
            env_name=task,
            robots=robots,
            controller_configs=controller_configs,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            control_freq=control_freq,
            horizon=episode_length,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=obs_cameras,
            camera_heights=image_height,
            camera_widths=image_width,
            camera_depths=camera_depths,
            reward_shaping=reward_shaping,
            **kwargs,
        )

        self.modders = dict.fromkeys(["texture", "color", "camera", "light"])
        if custom_reset_config is not None:
            if isinstance(custom_reset_config, str):
                custom_reset_config = ALL_PRESET_ARGUMENTS.get(
                    custom_reset_config, None
                )
            assert set(custom_reset_config.keys()).issubset(
                {
                    "xml_string",
                    "custom_texture",
                    "custom_color",
                    "custom_camera",
                    "custom_light",
                }
            ), "The keys of custom_reset_config must be the arguments to self.reset()"
            self._reset_config = custom_reset_config
            xml_string = custom_reset_config.get("xml_string")
            self._initialize_xml_modder(custom_reset_config.get("custom_texture", None))
            self.modder_seeds = dict.fromkeys(["color", "camera", "light"])
            self._initialize_modders(
                custom_color=custom_reset_config.get("custom_color", None),
                custom_camera=custom_reset_config.get("custom_camera", None),
                custom_light=custom_reset_config.get("custom_light", None),
            )
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
                    "custom_color",
                    "custom_camera",
                    "custom_light",
                ]
            )
            self.reset_xml_next = False

        obs_dict = self.env.observation_spec()
        self.observation_space = get_obs_shape_from_dict(self._reformat_obs(obs_dict))
        low, high = self.env.action_spec
        self.action_space = Box(low=low, high=high)

    def _initialize_xml_modder(self, custom_texture: Optional[Union[str, dict]]):
        if custom_texture is not None:
            if isinstance(custom_texture, str):
                custom_texture = ALL_TEXTURE_PRESETS.get(custom_texture, None)
            config = custom_texture.copy()
            if custom_texture.get("tex_to_change", None) is None:
                config["tex_to_change"] = DEFAULT_TASK_TEXTURE_LIST[self.task]
            self.modders["texture"] = XMLTextureModder(**config)

    def _initialize_modders(
        self,
        custom_color: Optional[Union[str, dict]],
        custom_camera: Optional[Union[str, dict]],
        custom_light: Optional[Union[str, dict]],
    ):
        if custom_color is not None:
            if isinstance(custom_color, str):
                custom_color = ALL_COLOR_PRESETS[custom_color]
            config = custom_color.copy()
            seed = config.pop("seed", None)
            self.modder_seeds["color"] = seed
            self.modders["color"] = TextureModder(self.env.sim, **config)
        if custom_camera is not None:
            if isinstance(custom_camera, str):
                custom_camera = ALL_CAMERA_PRESETS[custom_camera]
            config = custom_camera.copy()
            seed = config.pop("seed", None)
            self.modder_seeds["camera"] = seed
            self.modders["camera"] = CameraModder(self.env.sim, **config)
        if custom_light is not None:
            if isinstance(custom_light, str):
                custom_light = ALL_LIGHTING_PRESETS[custom_light]
            config = custom_light.copy()
            seed = config.pop("seed", None)
            self.modder_seeds["light"] = seed
            self.modders["light"] = LightingModder(self.env.sim, **config)

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
        custom_color: Optional[Union[str, dict]] = None,
        custom_camera: Optional[Union[str, dict]] = None,
        custom_light: Optional[Union[str, dict]] = None,
    ):
        """
        Reset the current environment. Additional options are used to alter the
        texture/camera/light of the environment.

        Args:
            xml_string (str or list): a Mujoco xml string or a list of Mujoco xml
                    strings. Check http://www.mujoco.org/book/XMLreference.html
                    for XML reference. The common use case is to get env's mujoco xml
                    via env.get_mujoco_xml(), modify it via the class CustomMujocoXML,
                    and pass CustomMujocoXML.to_string() as argument. If a list of
                    strings is used, one of them will be randomly selected.
            custom_texture (str or dict): a dictionary to be flattened as arguments to
                    generate a custom_xml.XMLTextureModder instance.
            custom_color (str or dict): a dictionary to be flattened as arguments to
                    generate a robosuite.utils.mjmod.TextureModder instance.
            custom_camera (str or dict): a dictionary to be flattened as arguments to
                    generate a robosuite.utils.mjmod.TextureModder instance.
            custom_light (str or dict): a dictionary to be flattened as arguments to
                    generate a robosuite.utils.mjmod.TextureModder instance.

                For custom_texture, custom_color, custom_camera and custom_light,
                Users can pass in a random seed to generate a RandomState instance for
                the corresponding Modder. Refer to the documentation of these modders
                for more documentations.
        """
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
        if custom_color is not None:
            if custom_color != self._reset_config.get("custom_color", None):
                self._reset_config["custom_color"] = copy.deepcopy(custom_color)
            else:
                custom_color = None
        if custom_camera is not None:
            if custom_camera != self._reset_config.get("custom_camera", None):
                self._reset_config["custom_camera"] = copy.deepcopy(custom_camera)
            else:
                custom_camera = None
        if custom_light is not None:
            if custom_light != self._reset_config.get("custom_light", None):
                self._reset_config["custom_light"] = copy.deepcopy(custom_light)
            else:
                custom_light = None

        # reset from xml should only be called if a different texture/xml is requested
        reset_from_xml = (
            xml_string is not None or custom_texture is not None or self.reset_xml_next
        )
        if reset_from_xml:
            self._reset_from_xml(xml_string)
            self.env.deterministic_reset = False
        else:
            self.env.reset()
        self.reset_xml_next = False

        if reset_from_xml or self._hard_reset:
            # sim has changed -> need to re-initialize modders if available
            self._initialize_modders(
                custom_color=self._reset_config.get("custom_color", None),
                custom_camera=self._reset_config.get("custom_camera", None),
                custom_light=self._reset_config.get("custom_light", None),
            )
        else:
            self._initialize_modders(
                custom_color=custom_color,
                custom_camera=custom_camera,
                custom_light=custom_light,
            )
        for modder_key in ["color", "camera", "light"]:
            modder = self.modders[modder_key]
            if modder:
                seed = self.modder_seeds[modder_key]
                if seed is None:
                    random_state = np.random.mtrand._rand
                else:
                    random_state = np.random.RandomState(seed)
                modder.random_state = random_state
                modder.randomize()
        # self.env.placement_initializer.reset()  # TODO gvrlb
        # self.env.placement_initializer.add_objects(self.env.door)
        # yang = self._reformat_obs(self.env._get_observations())
        return self._reformat_obs(self.env._get_observations())

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

        if self.modders["texture"] is not None:
            self.modders["texture"].random_texture_change(mujoco_xml)
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
