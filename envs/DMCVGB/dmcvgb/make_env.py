from dmcvgb.utils import omegaconf_to_dict
import hydra
from hydra import compose, initialize
import os
import dmc2gym
from dmcvgb import utils
from dmcvgb.wrappers import ColorWrapper, VideoWrapper, FrameStack

# def make_env(
#         domain_name,
#         task_name,
#         seed=0,
#         episode_length=1000,
#         frame_stack=3,
#         action_repeat=4,
#         image_size=100,
#         mode='train',
#         intensity=0.5
# ):
#     """Make environment for experiments"""
#     assert mode in {'train', 'color_easy', 'color_hard', 'video_easy', 'video_hard', 'distracting_cs'}, \
#         f'specified mode "{mode}" is not supported'
#
#     paths = []
#     is_distracting_cs = mode == 'distracting_cs'
#     if is_distracting_cs:
#         from distracting_control import suite as dc_suite
#         # import distracting_control.suite as dc_suite
#         loaded_paths = [os.path.join(dir_path, 'DAVIS/JPEGImages/480p') for dir_path in utils.load_config('datasets')]
#         for path in loaded_paths:
#             if os.path.exists(path):
#                 paths.append(path)
#     env = dmc2gym.make(
#         domain_name=domain_name,
#         task_name=task_name,
#         seed=seed,
#         visualize_reward=False,
#         from_pixels=True,
#         height=image_size,
#         width=image_size,
#         episode_length=episode_length,
#         frame_skip=action_repeat,
#         is_distracting_cs=is_distracting_cs,
#         distracting_cs_intensity=intensity,
#         background_dataset_paths=paths,
#         camera_id=1
#     )
#     if not is_distracting_cs:
#         env = VideoWrapper(env, mode, seed)
#     env = FrameStack(env, frame_stack)
#     if not is_distracting_cs:
#         env = ColorWrapper(env, mode, seed)
#
#     return env


def make_env(domain_name, task_name, seed, action_repeat=2, frame_stack=3, type='original', difficulty='easy'):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(config_path="../cfg"):
        cfg = compose(config_name="config")
        cfg_dict = omegaconf_to_dict(cfg)
    # assert cfg_dict['mode'] in {'train', 'color_easy', 'color_hard', 'video_easy', 'video_hard', 'distracting_cs'}, \
    #     f'specified mode "{cfg_dict["mode"]}" is not supported'
    cfg_dict['taskdef']['domain_name'] = domain_name
    cfg_dict['taskdef']['task_name'] = task_name
    cfg_dict['seed'] = seed
    cfg_dict['frame_stack'] = frame_stack
    cfg_dict['action_repeat'] = action_repeat
    cfg_dict['background']['type'] = type
    cfg_dict['background']['difficulty'] = difficulty

    paths = []
    # is_distracting_cs = cfg_dict['mode'] == 'distracting_cs'
    if cfg_dict['is_distracting_cs']:
        from distracting_control import suite as dc_suite
        # import distracting_control.suite as dc_suite
        loaded_paths = [os.path.join(dir_path, 'DAVIS/JPEGImages/480p') for dir_path in utils.load_config('datasets')]
        for path in loaded_paths:
            if os.path.exists(path):
                paths.append(path)
    env = dmc2gym.make(
        domain_name=cfg_dict['taskdef']['domain_name'],
        task_name=cfg_dict['taskdef']['task_name'],
        seed=cfg_dict['seed'],
        visualize_reward=False,
        from_pixels=True,
        height=cfg_dict['image_size'],
        width=cfg_dict['image_size'],
        episode_length=cfg_dict['episode_length'],
        frame_skip=cfg_dict['action_repeat'],
        is_distracting_cs=cfg_dict['is_distracting_cs'],
        distracting_cs_intensity=cfg_dict['intensity'],
        background_dataset_paths=paths,
        camera_id=cfg_dict['cam_id'],
        cam_pos=cfg_dict['cam_pos'],
    )

    if not cfg_dict['is_distracting_cs']:
        env = VideoWrapper(env, cfg_dict['background'], cfg_dict['seed'], cfg_dict['objects_color'], cfg_dict['cam_pos'])
    env = FrameStack(env, cfg_dict['frame_stack'])
    if not cfg_dict['is_distracting_cs']:
        env = ColorWrapper(env, cfg_dict['background'], cfg_dict['seed'], cfg_dict['objects_color'], cfg_dict['table_texture'], cfg_dict['light_position'], cfg_dict['light_color'], cfg_dict['moving_light'], cfg_dict['cam_pos'])

    return env
