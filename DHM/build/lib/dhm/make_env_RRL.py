import os
import gc
from copy import deepcopy
import dm_env
from dhm.utils import omegaconf_to_dict
from hydra import compose, initialize
from dhm.modify_mjcf import *
from dhm.utils import GymEnv
from dm_env import StepType, specs
from typing import Any, NamedTuple
import gym
from abc import ABC
from PIL import Image
import numpy as np
import torch
import cv2
from dhm.rrl_encoder import Encoder, IdentityEncoder
from collections import deque


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ExtendedTimeStepAdroit(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    observation_sensor: Any
    action: Any
    n_goal_achieved: Any
    time_limit_reached: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


_mj_envs = {'pen-adroit-v0', 'hammer-adroit-v0', 'door-adroit-v0', 'relocate-adroit-v0',
            'pen-mpl-v0', 'hammer-mpl-v0', 'door-mpl-v0', 'relocate-mpl-v0'}


## DHMEnv 解析配置文件，修改mjcf  按照规范提供接口（dmc_env之类）
## BasicDHMEnv 提供step、reset、obs，不必在意接口
def make_encoder(encoder, encoder_type, device, is_eval=True):
    if not encoder:
        if encoder_type == 'resnet34' or encoder_type == 'resnet18':
            encoder = Encoder(encoder_type)
        elif encoder_type == 'identity':
            encoder = IdentityEncoder()
        else:
            print("Please enter valid encoder_type.")
            raise Exception
    if is_eval:
        encoder.eval()
    encoder.to(device)
    return encoder


class BasicDHMEnv(gym.Env, ABC):
    def __init__(self, env, feature_type, cfg_dict, cameras, latent_dim=512, hybrid_state=True, channels_first=False,
                 height=84, width=84, test_image=False, num_repeats=1, num_frames=1, encoder_type=None, device=None):
        self.step_count = 0
        self.cfg_dict = cfg_dict
        self._env = env
        self.env_id = env.env.unwrapped.spec.id
        self.device = device
        self._num_repeats = num_repeats
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self.random_state = np.random.RandomState(cfg_dict['task']['seed'])
        self.encoder = None
        self.transforms = None
        self.encoder_type = encoder_type
        if encoder_type is not None:
            self.encoder = make_encoder(encoder=None, encoder_type=self.encoder_type, device=self.device, is_eval=True)
            self.transforms = self.encoder.get_transform()

        if test_image:
            print("======================adroit image test mode==============================")
            print("======================adroit image test mode==============================")
            print("======================adroit image test mode==============================")
            print("======================adroit image test mode==============================")
        self.test_image = test_image

        self.cam_list = cameras
        self.latent_dim = latent_dim
        self.hybrid_state = hybrid_state
        self.channels_first = channels_first
        self.height = height
        self.width = width

        self.feature_type = feature_type
        # self.cam_list = cfg_dict['task']['cam_list']
        # self.channels_first = cfg_dict['channels_first']
        # self.height = cfg_dict['image_height']
        # self.width = cfg_dict['image_width']

        self.action_space = self._env.action_space
        self.env_kwargs = {'cameras': cameras, 'latent_dim': latent_dim, 'hybrid_state': hybrid_state,
                           'channels_first': channels_first, 'height': height, 'width': width}
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=[3, self.width, self.height], dtype=np.uint8
        )
        self.sim = env.env.sim
        # self._env.spec.observation_dim = self.height * self.width * len(self.cam_list)
        self._env.spec.observation_dim = latent_dim

        if hybrid_state:
            if self.env_id in _mj_envs:
                self._env.spec.observation_dim += 24  # Assuming 24 states for adroit hand. TODO

        self.spec = self._env.spec
        self.observation_dim = self.spec.observation_dim
        self.horizon = self._env.env.spec.max_episode_steps

        current_dir = os.path.dirname(__file__)
        self.color_index = 0
        # if self.cfg_dict["task"]["background"]['type'] == 'color' or self.cfg_dict["task"]["objects_color"] != 'original':
        self.colors = {'easy': torch.load(current_dir + '/' + f'../mj_envs/dependencies/color/easy.pt'),
                       'hard': torch.load(current_dir + '/' + f'../mj_envs/dependencies/color/hard.pt')}
        if self.cfg_dict["task"]["background"]['type'] == 'video':
            video_dir = os.path.join(current_dir + '/' + '../mj_envs/dependencies/video',
                                     self.cfg_dict['task']['background']['difficulty'])
            self.num_video = len(os.listdir(video_dir))
            self.video_paths = [os.path.join(video_dir, f'video{i}.mp4') for i in range(self.num_video)]
            self.video_index = 0
        if self.cfg_dict['task']['moving_light'] != 'original':
            self.moving_light_step = self.cfg_dict['setting']['moving_light'][self.cfg_dict['task']['moving_light']][
                'step']
            self.moving_light_range = self.cfg_dict['setting']['moving_light'][self.cfg_dict['task']['moving_light']][
                'range']

        # yang = self._env.env.env.model
        self.origin_cam_pos = deepcopy(self._env.env.env.model.cam_pos)
        self.origin_cam_fovy = deepcopy(self._env.env.env.model.cam_fovy)
        self.origin_cam_quat = deepcopy(self._env.env.env.model.cam_quat)
        self.origin_light_pos = deepcopy(self._env.env.env.model.light_pos[:])
        # self.origin_light_specular = deepcopy(self._env.env.env.model.light_specular)

    # TODO util
    def load_video(self):
        self.video_index = (self.video_index + 1) % self.num_video
        video_path = self.video_paths[self.video_index]
        cap = cv2.VideoCapture(video_path)
        self.video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        buf = np.empty(
            (self.video_len, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3),
            np.dtype('uint8'))
        i, ret = 0, True
        while (i < self.video_len and ret):
            ret, frame = cap.read()
            buf[i] = frame
            i += 1
        cap.release()
        return np.moveaxis(buf, -1, 1)  # (500, 3, 352, 352)

    def get_obs(self):
        if self.env_id in _mj_envs:
            env_state = self._env.env.get_env_state()
            qp = env_state['qpos']

        if self.env_id == 'pen-adroit-v0':
            qp = qp[:-6]
        elif self.env_id == 'door-adroit-v0':
            qp = qp[4:-2]
        elif self.env_id == 'hammer-adroit-v0':
            qp = qp[2:-7]
        elif self.env_id == 'relocate-adroit-v0':
            qp = qp[6:-6]

        imgs = []  # number of image is number of camera

        if self.encoder is not None:
            for cam in self.cam_list:
                img = self._env.env.sim.render(width=self.width, height=self.height, mode='offscreen', camera_name=cam,
                                               device_id=0)
                # img = env.env.sim.render(width=84, height=84, mode='offscreen')
                img = img[::-1, :, :]  # Image given has to be flipped
                if self.channels_first:
                    img = img.transpose((2, 0, 1))

                assert self.cfg_dict['task']['background']['type'] in ('original', 'color',
                                                                       'video'), 'self.cfg_dict[\'background\'][\'type\'] must in (\'original\', \'color\', \'video\')'
                if self.cfg_dict['task']['background']['type'] == 'video' and self.cfg_dict['task']['mode'] == 'test':
                    #  TODO change the background by alternating pixels belong to background
                    # fixedRGB = np.array([110, 52, 128], dtype='uint8')
                    fixedRGB = np.array([107, 51, 127], dtype='uint8')
                    img = img.transpose(1, 2, 0)
                    # idx = np.where((img[:][:] == fixedRGB).all(axis=2), 1, 0)
                    idx = np.where((img[:][:] == fixedRGB).all(axis=2), np.array(1, dtype=np.uint8),
                                   np.array(0, dtype=np.uint8))
                    idx = np.array([idx, idx, idx])
                    img = img.transpose(2, 0, 1)
                    img = (1 - idx) * img + idx * self.video_buf[self.step_count % self.video_len, :,
                                                  :self.cfg_dict['image_height'], :self.cfg_dict['image_width']]
                elif self.cfg_dict['task']['background']['type'] == 'color':
                    pass
                self.step_count += 1

                # img = img.astype(np.uint8)
                img = Image.fromarray(img)
                img = self.transforms(img)
                imgs.append(img)

            inp_img = torch.stack(imgs).to(self.device)  # [num_cam, C, H, W]
            z = self.encoder.get_features(inp_img).reshape(-1)
            # assert z.shape[0] == self.latent_dim, "Encoded feature length : {}, Expected : {}".format(z.shape[0], self.latent_dim)
            pixels = z
        else:
            if not self.test_image:
                for cam in self.cam_list:  # for each camera, render once
                    img = self._env.env.sim.render(width=self.width, height=self.height, mode='offscreen',
                                                   camera_name=cam, device_id=0)  # TODO device id will think later
                    # img = img[::-1, :, : ] # Image given has to be flipped
                    if self.channels_first:
                        img = img.transpose((2, 0, 1))  # then it's 3 x width x height

                    assert self.cfg_dict['task']['background']['type'] in ('original', 'color',
                                                                           'video'), 'self.cfg_dict[\'background\'][\'type\'] must in (\'original\', \'color\', \'video\')'
                    if self.cfg_dict['task']['background']['type'] == 'video' and self.cfg_dict['task'][
                        'mode'] == 'test':
                        #  TODO change the background by alternating pixels belong to background
                        # fixedRGB = np.array([110, 52, 128], dtype='uint8')
                        fixedRGB = np.array([107, 51, 127], dtype='uint8')
                        img = img.transpose(1, 2, 0)
                        # idx = np.where((img[:][:] == fixedRGB).all(axis=2), 1, 0)
                        idx = np.where((img[:][:] == fixedRGB).all(axis=2), np.array(1, dtype=np.uint8),
                                       np.array(0, dtype=np.uint8))
                        idx = np.array([idx, idx, idx])
                        img = img.transpose(2, 0, 1)
                        img = (1 - idx) * img + idx * self.video_buf[self.step_count % self.video_len, :,
                                                      :self.cfg_dict['image_height'], :self.cfg_dict['image_width']]
                    elif self.cfg_dict['task']['background']['type'] == 'color':
                        pass
                    self.step_count += 1
                    # we should do channels first... (not sure why by default it's not, maybe they did some transpose when using the encoder?)
                    # img = img.astype(np.uint8)
                    # img = Image.fromarray(img) # TODO is this necessary?
                    imgs.append(img)
            else:
                img = (np.random.rand(1, 84, 84) * 255).astype(np.uint8)
                imgs.append(img)
            pixels = np.concatenate(imgs, axis=0)

        if not self.hybrid_state:  # this defaults to True... so RRL uses hybrid state
            qp = None
        return pixels, qp

    def get_env_infos(self):
        return self._env.get_env_infos()

    def set_seed(self, seed):
        return self._env.set_seed(seed)

    def get_stacked_pixels(self):  # TODO fix it
        assert len(self._frames) == self._num_frames
        stacked_pixels = np.concatenate(list(self._frames), axis=0)
        return stacked_pixels

    def reset(self):
        if (self.feature_type == 'pixel' or self.feature_type == 'hybrid') and self.cfg_dict['task']['mode'] == 'test':
            self.step_count = 0
            color = None
            self.color_index = (self.color_index + 1) % len(self.colors['easy'])
            if self.cfg_dict["task"]["objects_color"] != 'original':
                color = self.colors[self.cfg_dict['task']['objects_color']][
                    self.random_state.randint(len(self.colors[self.cfg_dict['task']['objects_color']]))]['self_rgb']
            src_mjcf_path = f'../mj_envs/mj_envs/hand_manipulation_suite/assets/{self.cfg_dict["task"]["taskdef"]["hand"]}/DAPG_assets_template.xml'
            dest_mjcf_path = f'../mj_envs/mj_envs/hand_manipulation_suite/assets/{self.cfg_dict["task"]["taskdef"]["hand"]}/DAPG_assets_test_{str(os.getpid())}.xml'
            modify_assets(src_mjcf_path, dest_mjcf_path, color, self.cfg_dict["task"]['table_texture'],
                          self.color_index % 10)
            src_mjcf_path = f'../mj_envs/mj_envs/hand_manipulation_suite/assets/{self.cfg_dict["task"]["taskdef"]["hand"]}/DAPG_{self.cfg_dict["task"]["taskdef"]["task"]}_template.xml'
            dest_mjcf_path = f'../mj_envs/mj_envs/hand_manipulation_suite/assets/{self.cfg_dict["task"]["taskdef"]["hand"]}/DAPG_{self.cfg_dict["task"]["taskdef"]["task"]}_test_{str(os.getpid())}.xml'
            modify_mjcf_via_dict(src_mjcf_path, dest_mjcf_path, self.cfg_dict['task'], self.random_state,
                                 self.cfg_dict['setting'])
            del self._env
            gc.collect()
            self._env = GymEnv(self.cfg_dict['task']['env_name'])

            if self.cfg_dict['task']['background']['type'] == 'video':
                self.video_buf = self.load_video()
            elif self.cfg_dict['task']['background']['type'] == 'color':
                self._env = GymEnv(self.cfg_dict['task']['env_name'])
                color = self.colors[self.cfg_dict['task']['background']['difficulty']][
                    self.random_state.randint(len(self.colors[self.cfg_dict['task']['background']['difficulty']]))][
                    'skybox_rgb']
                self._env.env.env.model.tex_rgb[0:1280000:3] = color[0] * 255
                self._env.env.env.model.tex_rgb[1:1280000:3] = color[1] * 255
                self._env.env.env.model.tex_rgb[2:1280000:3] = color[2] * 255

            if self.cfg_dict['task']['camera']['position'] != 'original':
                self._env.env.env.model.cam_pos[:, 0] = self.origin_cam_pos[:, 0] + self.random_state.uniform(
                    self.cfg_dict['setting']['camera']['position'][self.cfg_dict['task']['camera']['position']]['x'][0],
                    self.cfg_dict['setting']['camera']['position'][self.cfg_dict['task']['camera']['position']]['x'][1])
                self._env.env.env.model.cam_pos[:, 1] = self.origin_cam_pos[:, 1] + self.random_state.uniform(
                    self.cfg_dict['setting']['camera']['position'][self.cfg_dict['task']['camera']['position']]['y'][0],
                    self.cfg_dict['setting']['camera']['position'][self.cfg_dict['task']['camera']['position']]['y'][1])
                self._env.env.env.model.cam_pos[:, 2] = self.origin_cam_pos[:, 2] + self.random_state.uniform(
                    self.cfg_dict['setting']['camera']['position'][self.cfg_dict['task']['camera']['position']]['z'][0],
                    self.cfg_dict['setting']['camera']['position'][self.cfg_dict['task']['camera']['position']]['z'][1])
            if self.cfg_dict['task']['camera']['orientation'] != 'original':
                self._env.env.env.model.cam_quat[:, 0] = self.origin_cam_quat[:, 0] + self.random_state.uniform(
                    self.cfg_dict['setting']['camera']['orientation'][self.cfg_dict['task']['camera']['orientation']][
                        'a'][0],
                    self.cfg_dict['setting']['camera']['orientation'][self.cfg_dict['task']['camera']['orientation']][
                        'a'][1])
                self._env.env.env.model.cam_quat[:, 1] = self.origin_cam_quat[:, 1] + self.random_state.uniform(
                    self.cfg_dict['setting']['camera']['orientation'][self.cfg_dict['task']['camera']['orientation']][
                        'b'][0],
                    self.cfg_dict['setting']['camera']['orientation'][self.cfg_dict['task']['camera']['orientation']][
                        'b'][1])
                self._env.env.env.model.cam_quat[:, 2] = self.origin_cam_quat[:, 2] + self.random_state.uniform(
                    self.cfg_dict['setting']['camera']['orientation'][self.cfg_dict['task']['camera']['orientation']][
                        'c'][0],
                    self.cfg_dict['setting']['camera']['orientation'][self.cfg_dict['task']['camera']['orientation']][
                        'c'][1])
                self._env.env.env.model.cam_quat[:, 3] = self.origin_cam_quat[:, 3] + self.random_state.uniform(
                    self.cfg_dict['setting']['camera']['orientation'][self.cfg_dict['task']['camera']['orientation']][
                        'd'][0],
                    self.cfg_dict['setting']['camera']['orientation'][self.cfg_dict['task']['camera']['orientation']][
                        'd'][1])
            if self.cfg_dict['task']['camera']['fov'] != 'original':
                self._env.env.env.model.cam_fovy[:] = self.origin_cam_fovy[:] + self.random_state.randint(
                    self.cfg_dict['setting']['camera']['fov'][self.cfg_dict['task']['camera']['fov']][0],
                    self.cfg_dict['setting']['camera']['fov'][self.cfg_dict['task']['camera']['fov']][1])

            if self.cfg_dict['task']['light']['position'] != 'original':
                self._env.env.env.model.light_pos[:, 0] = self.origin_light_pos[:, 0] + self.random_state.uniform(
                    self.cfg_dict['setting']['light']['position'][self.cfg_dict['task']['light']['position']]['x'][0],
                    self.cfg_dict['setting']['light']['position'][self.cfg_dict['task']['light']['position']]['x'][1])
                self._env.env.env.model.light_pos[:, 1] = self.origin_light_pos[:, 1] + self.random_state.uniform(
                    self.cfg_dict['setting']['light']['position'][self.cfg_dict['task']['light']['position']]['y'][0],
                    self.cfg_dict['setting']['light']['position'][self.cfg_dict['task']['light']['position']]['y'][1])
                self._env.env.env.model.light_pos[:, 2] = self.origin_light_pos[:, 2] + self.random_state.uniform(
                    self.cfg_dict['setting']['light']['position'][self.cfg_dict['task']['light']['position']]['z'][0],
                    self.cfg_dict['setting']['light']['position'][self.cfg_dict['task']['light']['position']]['z'][1])

        self._env.reset()
        pixels, obs_sensor = self.get_obs()
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        stacked_pixels = self.get_stacked_pixels()
        return stacked_pixels, obs_sensor

    def get_obs_for_first_state_but_without_reset(self):
        pixels, sensor_info = self.get_obs()
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        stacked_pixels = self.get_stacked_pixels()
        return stacked_pixels, sensor_info

    def step(self, action):
        if self.cfg_dict['task']['moving_light'] != 'original' and self.cfg_dict['task']['mode'] == 'test':
            if abs(self._env.env.env.model.light_pos[:, 1] - self.origin_light_pos[:, 1]) >= self.moving_light_range:
                # print(self._env.env.env.model.light_pos[:,1] - self.origin_light_pos[:,1])
                self.moving_light_step *= -1
            self._env.env.env.model.light_pos[:, 1] += self.moving_light_step
        reward_sum = 0.0
        discount_prod = 1.0  # TODO pen can terminate early
        n_goal_achieved = 0
        for i_action in range(self._num_repeats):
            obs, reward, done, env_info = self._env.step(action)
            reward_sum += reward
            if env_info['goal_achieved'] == True:
                n_goal_achieved += 1
            if done:
                break
        env_info['n_goal_achieved'] = n_goal_achieved
        # now get stacked frames
        pixels, sensor_info = self.get_obs()
        self._frames.append(pixels)
        stacked_pixels = self.get_stacked_pixels()
        return [stacked_pixels, sensor_info], reward_sum, done, env_info

    def set_env_state(self, state):
        return self._env.set_env_state(state)

    def get_env_state(self):
        return self._env.get_env_state()

    def evaluate_policy(self, policy,
                        num_episodes=5,
                        horizon=None,
                        gamma=1,
                        visual=False,
                        percentile=[],
                        get_full_dist=False,
                        mean_action=False,
                        init_env_state=None,
                        terminate_at_done=True,
                        seed=123):
        # TODO this needs to be rewritten

        self.set_seed(seed)
        horizon = self.horizon if horizon is None else horizon
        mean_eval, std, min_eval, max_eval = 0.0, 0.0, -1e8, -1e8
        ep_returns = np.zeros(num_episodes)
        self.encoder.eval()

        for ep in range(num_episodes):
            o = self.reset()
            if init_env_state is not None:
                self.set_env_state(init_env_state)
            t, done = 0, False
            while t < horizon and (done == False or terminate_at_done == False):
                self.render() if visual is True else None
                o = self.get_obs(self._env.get_obs())
                a = policy.get_action(o)[1]['evaluation'] if mean_action is True else policy.get_action(o)[0]
                o, r, done, _ = self.step(a)
                ep_returns[ep] += (gamma ** t) * r
                t += 1

        mean_eval, std = np.mean(ep_returns), np.std(ep_returns)
        min_eval, max_eval = np.amin(ep_returns), np.amax(ep_returns)
        base_stats = [mean_eval, std, min_eval, max_eval]

        percentile_stats = []
        for p in percentile:
            percentile_stats.append(np.percentile(ep_returns, p))

        full_dist = ep_returns if get_full_dist is True else None

        return [base_stats, percentile_stats, full_dist]

    def get_pixels_with_width_height(self, w, h):
        imgs = []  # number of image is number of camera

        for cam in self.cam_list:  # for each camera, render once
            img = self._env.env.sim.render(width=w, height=h, mode='offscreen', camera_name=cam,
                                           device_id=0)  # TODO device id will think later
            # img = img[::-1, :, : ] # Image given has to be flipped
            if self.channels_first:
                img = img.transpose((2, 0, 1))  # then it's 3 x width x height
            # we should do channels first... (not sure why by default it's not, maybe they did some transpose when using the encoder?)
            # img = img.astype(np.uint8)
            # img = Image.fromarray(img) # TODO is this necessary?
            imgs.append(img)

        pixels = np.concatenate(imgs, axis=0)
        return pixels


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    observation_sensor: Any
    action: Any

    # n_goal_achieved: Any
    # time_limit_reached: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    # def __getitem__(self, attr):
    #     print(attr)
    #     return getattr(self, attr)


class DHMEnv():  # dhm is for dexterous hand manipulation
    def __init__(self, cfg_dict, env_name, test_image=False, cam_list=None,
                 num_repeats=2, num_frames=3, env_feature_type='pixels', device=None, reward_rescale=False):

        cfg_dict['task']['env_name'] = cfg_dict['task']['taskdef']['task'] + '-' + cfg_dict['task']['taskdef'][
            'hand'] + '-' + cfg_dict['task']['taskdef']['version']
        self.env_name = cfg_dict['task']['env_name']

        self.cfg_dict = cfg_dict

        reward_rescale_dict = {
            'hammer-adroit-v0': 1 / 100,
            'door-adroit-v0': 1 / 20,
            'pen-adroit-v0': 1 / 50,
            'relocate-adroit-v0': 1 / 30,
        }
        if reward_rescale:
            self.reward_rescale_factor = reward_rescale_dict[self.env_name]
        else:
            self.reward_rescale_factor = 1
        self.random_state = np.random.RandomState(cfg_dict['task']['seed'])
        src_mjcf_path = f'../mj_envs/mj_envs/hand_manipulation_suite/assets/{cfg_dict["task"]["taskdef"]["hand"]}/DAPG_assets_template.xml'
        dest_mjcf_path = f'../mj_envs/mj_envs/hand_manipulation_suite/assets/{cfg_dict["task"]["taskdef"]["hand"]}/DAPG_assets_test_{str(os.getpid())}.xml'
        modify_assets(src_mjcf_path, dest_mjcf_path, None, 'original', None)
        # 根据cfg解析修改mjcf DAPG_hammer_template.xml->DAPG_hammer.xml
        src_mjcf_path = f'../mj_envs/mj_envs/hand_manipulation_suite/assets/{cfg_dict["task"]["taskdef"]["hand"]}/DAPG_{cfg_dict["task"]["taskdef"]["task"]}_template.xml'
        dest_mjcf_path = f'../mj_envs/mj_envs/hand_manipulation_suite/assets/{cfg_dict["task"]["taskdef"]["hand"]}/DAPG_{cfg_dict["task"]["taskdef"]["task"]}_test_{str(os.getpid())}.xml'
        # import pdb;pdb.set_trace()
        modify_mjcf_via_dict(src_mjcf_path, dest_mjcf_path, cfg_dict['task'], self.random_state, cfg_dict['setting'])

        default_env_to_cam_list = {
            'hammer-adroit-v0': ['top'],
            'door-adroit-v0': ['top'],
            'pen-adroit-v0': ['vil_camera'],
            'relocate-adroit-v0': ['cam1', 'cam2', 'cam3'],
        }
        if cfg_dict['task']['cam_list'] is None:
            cfg_dict['task']['cam_list'] = default_env_to_cam_list[self.env_name]

        env = GymEnv(self.env_name)
        if env_feature_type == 'state':
            raise NotImplementedError("state env not ready")
        elif env_feature_type == 'resnet18' or env_feature_type == 'resnet34':
            # TODO maybe we will just throw everything into it..
            height = cfg_dict['image_height']
            width = cfg_dict['image_width']
            latent_dim = 512
            env = BasicDHMEnv(env, cameras=cfg_dict['task']['cam_list'],
                              height=height, width=width, latent_dim=latent_dim, hybrid_state=True,
                              test_image=test_image, channels_first=False, num_repeats=num_repeats,
                              num_frames=num_frames, encoder_type=env_feature_type,
                              device=device, feature_type=cfg_dict['feature_type'], cfg_dict=cfg_dict
                              )
        elif env_feature_type == 'pixels':
            height = cfg_dict['image_height']
            width = cfg_dict['image_width']
            latent_dim = height * width * len(cfg_dict['task']['cam_list']) * num_frames
            # RRL class instance is environment wrapper...
            env = BasicDHMEnv(env, cameras=cfg_dict['task']['cam_list'],
                              height=height, width=width, latent_dim=latent_dim, hybrid_state=True,
                              test_image=test_image, channels_first=True, num_repeats=num_repeats,
                              num_frames=num_frames, device=device, feature_type=cfg_dict['feature_type'],
                              cfg_dict=cfg_dict)
        else:
            raise ValueError("env feature not supported")

        # env = BasicDHMEnv(env, feature_type=cfg_dict['feature_type'], cfg_dict=cfg_dict)

        self._env = env
        self.obs_dim = env.spec.observation_dim
        self.obs_sensor_dim = 24
        self.act_dim = env.spec.action_dim
        self.horizon = env.spec.horizon
        number_channel = len(cfg_dict['task']['cam_list']) * 3 * num_frames

        if env_feature_type == 'pixels':
            self._obs_spec = specs.BoundedArray(shape=(number_channel, 84, 84), dtype='uint8', name='observation',
                                                minimum=0, maximum=255)
            self._obs_sensor_spec = specs.Array(shape=(self.obs_sensor_dim,), dtype='float32',
                                                name='observation_sensor')
        elif env_feature_type == 'resnet18' or env_feature_type == 'resnet34':
            self._obs_spec = specs.Array(shape=(512 * num_frames * len(cam_list),), dtype='float32',
                                         name='observation')  # TODO fix magic number
            self._obs_sensor_spec = specs.Array(shape=(self.obs_sensor_dim,), dtype='float32',
                                                name='observation_sensor')
        self._action_spec = specs.BoundedArray(shape=(self.act_dim,), dtype='float32', name='action', minimum=-1.0,
                                               maximum=1.0)

    def reset(self):
        obs_pixels, obs_sensor = self._env.reset()
        obs_sensor = obs_sensor.astype(np.float32)
        action_spec = self.action_spec()
        action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        time_step = ExtendedTimeStepAdroit(observation=obs_pixels,
                                           observation_sensor=obs_sensor,
                                           step_type=StepType.FIRST,
                                           action=action,
                                           reward=0.0,
                                           discount=1.0,
                                           n_goal_achieved=0,
                                           time_limit_reached=False)
        return time_step

    def get_current_obs_without_reset(self):
        # use this to obtain the first state in a demo
        obs_pixels, obs_sensor = self._env.get_obs_for_first_state_but_without_reset()
        obs_sensor = obs_sensor.astype(np.float32)
        action_spec = self.action_spec()
        action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        time_step = ExtendedTimeStepAdroit(observation=obs_pixels,
                                           observation_sensor=obs_sensor,
                                           step_type=StepType.FIRST,
                                           action=action,
                                           reward=0.0,
                                           discount=1.0,
                                           n_goal_achieved=0,
                                           time_limit_reached=False)
        return time_step

    def get_pixels_with_width_height(self, w, h):
        return self._env.get_pixels_with_width_height(w, h)

    def step(self, action, force_step_type=None, debug=False):
        obs_all, reward, done, env_info = self._env.step(action)
        obs_pixels, obs_sensor = obs_all
        obs_sensor = obs_sensor.astype(np.float32)

        discount = 1.0
        n_goal_achieved = env_info['n_goal_achieved']
        time_limit_reached = env_info['TimeLimit.truncated'] if 'TimeLimit.truncated' in env_info else False
        if done:
            steptype = StepType.LAST
        else:
            steptype = StepType.MID

        if done and not time_limit_reached:
            discount = 0.0

        if force_step_type is not None:
            if force_step_type == 'mid':
                steptype = StepType.MID
            elif force_step_type == 'last':
                steptype = StepType.LAST
            else:
                steptype = StepType.FIRST

        reward = reward * self.reward_rescale_factor

        time_step = ExtendedTimeStepAdroit(observation=obs_pixels,
                                           observation_sensor=obs_sensor,
                                           step_type=steptype,
                                           action=action,
                                           reward=reward,
                                           discount=discount,
                                           n_goal_achieved=n_goal_achieved,
                                           time_limit_reached=time_limit_reached)

        if debug:
            return obs_all, reward, done, env_info
        return time_step

    def observation_spec(self):
        return self._obs_spec

    def observation_sensor_spec(self):
        return self._obs_sensor_spec

    def action_spec(self):
        return self._action_spec

    def set_env_state(self, state):
        self._env.set_env_state(state)

    def del_xml(self):
        current_dir = os.path.dirname(__file__)
        os.chdir(current_dir)
        dir_path = f"../mj_envs/mj_envs/hand_manipulation_suite/assets/{self.cfg_dict['task']['taskdef']['hand']}/"
        files = os.listdir(dir_path)
        for file in files:
            if f'test_{str(os.getpid())}' in file:
                os.remove(os.path.join(dir_path, file))




    # def __getattr__(self, name):
    #     return getattr(self, name)


# they create an adroit env by initialize a 'AdroitEnv' in RRL or VRL3.
# We create an env with the same API by call make_env(), just pass the same args
# env_name is {task}-v0
import hydra


def make_env_RRL(env_name, test_image=False, cam_list=None,
                 num_repeats=2, num_frames=3, env_feature_type='pixels', device=None, reward_rescale=False, mode='train', seed=1):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    origin_dir = os.getcwd()
    current_dir = os.path.dirname(__file__)
    os.chdir(current_dir)
    task = env_name.split("-")[0]
    with initialize(config_path="../cfg"):
        cfg = compose(config_name="config", overrides=[f"task={task}"])
        cfg_dict = omegaconf_to_dict(cfg)
    cfg_dict['task']['mode'] = mode
    cfg_dict['task']['seed'] = seed
    print(f"Now the mode is {cfg_dict['task']['mode']}")
    # dir_path = f"../mj_envs/mj_envs/hand_manipulation_suite/assets/{cfg_dict['task']['taskdef']['hand']}/"
    # files = os.listdir(dir_path)
    # for file in files:
    #     if 'test' in file:
    #         os.remove(os.path.join(dir_path, file))
    env = DHMEnv(cfg_dict, env_name, test_image, cam_list, num_repeats, num_frames, env_feature_type, device,
                 reward_rescale)
    os.chdir(origin_dir)
    return env


if __name__ == '__main__':
    env = make_env_RRL('hammer')
