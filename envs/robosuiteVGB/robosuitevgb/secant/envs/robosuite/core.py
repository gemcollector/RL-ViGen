import robosuite as suite
from typing import Union, List, Optional, Dict
from .adapter import RobosuiteAdapter
from robosuite.utils.mjcf_utils import ALL_TEXTURES, TEXTURES
from secant.wrappers import TimeLimit, SingleModality, FrameStack


__all__ = ["make_robosuite", "ALL_TASKS", "ALL_ROBOTS", "ALL_TEXTURES", "TEXTURES"]


ALL_TASKS = ["Door", "TwoArmPegInHole", "NutAssemblyRound", "TwoArmLift"]
ALL_ROBOTS = list(suite.ALL_ROBOTS)


def make_robosuite(
    task: str,
    robots: Union[str, List[str]] = "Panda",
    controller_configs: Optional[Union[Dict, List[Dict]]] = None,
    controller_types: Optional[Union[str, List[str]]] = "OSC_POSE",
    headless: bool = True,
    obs_modality: Optional[List[str]] = ["rgb"],
    render_camera: str = "frontview",
    control_freq: int = 20,
    episode_length: Optional[int] = 500,
    ignore_done: bool = False,
    hard_reset: bool = True,  # TODO gvrlb
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
    single_modality_obs: bool = True,
    frame_stack: Optional[int] = 3,
    **kwargs,
):
    """
    Low-level make

    Args:
        task (str): Task name. Check ALL_TASK for a full list.
        robots (str or List[str]): Specification for specific robot arm(s) to be
                instantiated within this env. Check ALL_ROBOTS for a full list.
        controller_configs (str or list of dict): If set, contains relevant controller
                parameters for creating a custom controller. Else, uses the default
                controller for this specific task. Should either be single dict if
                same controller is to be used for all robots or else it should be a
                list of the same length as "robots" param.
        controller_types (str or list of str): Set which controller_config to use.
                Either "OSC_POSE" or "JOINT_VELOCITY".
                If set, controller_configs will be overridden.
        headless (bool): Whether to turn on headless mode.
        obs_modality (List[str]): Should be subset of ["rgb", "state"]. "rgb" turns on
                pixel observation and "state" turns on robot state observation.
        single_modality_obs (bool): Whether to use single-modality obs (if applicable)
        render_camera (str): name of camera to use when calling env.render(). May only
                work if this camera is included in obs_cameras.
        control_freq (int): how many control signals to receive in every simulated
                second. This sets the amount of simulation time that passes between
                every action input.
        episode_length (int): Every episode lasts for exactly @episode_length timesteps.
                If None, episode lasts forever.
        ignore_done (bool): True if never terminating the environment (ignore @episode_length).
        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset
                call, else, only calls sim.reset and resets all robosuite-internal
                variables. [IMPORTANT NOTE: avoid setting hard_reset=True when using
                customization unless necessary]
        obs_cameras (str or List[str]): name of cameras to be rendered as image
                observation. At least one camera must be specified if "rgb" is in
                @obs_modality.
                Note: To render all robots' cameras of a certain type
                    (e.g.: "robotview" or "eye_in_hand"), use the convention
                    "all-{name}" (e.g.: "all-robotview") to automatically render
                    all camera images from each robot's camera list).
        frame_stack (int or None): specifies the number of frames to be stacked.
        channel_first (bool): set True -> observation shape (channel, height, width);
                set False -> observation shape (height, width, channel).
        image_height (int): height of camera frame.
        image_width (int): width of camera frame.
        camera_depths (bool): True if rendering RGB-D, and RGB otherwise.
        custom_reset_config (str or dict): a dictionary of arguments that will be passed
                to RobosuiteAdapter.reset(). The keys of this dictionary should be
                subset of {custom_texture, custom_color, custom_camera, custom_light}.
                Check the reset() method in adapter.py for more documentations.
        mode (str): "train", "eval-easy", "eval-medium" or "eval-hard". This is used with scene_id.
        scene_id (int or None): If mode == "train", scene_id controls which domain randomization
                scene to use (scene_id=0 indicates using the original train scene).
                If mode == "eval-easy", "eval-medium" or "eval-hard", scene_id controls which eval scene to use.
        reward_shaping (bool): If True, use dense rewards.
        kwargs: Other arguments to pass to the Robosuite environments.
    """

    assert channel_first, "we only support channel_first=True"
    assert task in ALL_TASKS, f"Task {task} does not exist"

    env = RobosuiteAdapter(
        task=task,
        robots=robots,
        controller_configs=controller_configs,
        controller_types=controller_types,
        headless=headless,
        obs_modality=obs_modality,
        render_camera=render_camera,
        control_freq=control_freq,
        episode_length=episode_length,
        ignore_done=ignore_done,
        hard_reset=hard_reset,
        obs_cameras=obs_cameras,
        channel_first=channel_first,
        image_height=image_height,
        image_width=image_width,
        camera_depths=camera_depths,
        custom_reset_config=custom_reset_config,
        mode=mode,
        scene_id=scene_id,
        reward_shaping=reward_shaping,
        verbose=verbose,
        **kwargs,
    )

    env = TimeLimit(env, max_episode_steps=episode_length)
    if single_modality_obs and len(obs_modality) == 1:
        env = SingleModality(env, obs_modality[0])
    if frame_stack and frame_stack > 0:
        env = FrameStack(env, frame_stack)
    return env
