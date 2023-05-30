import os
import gc
from copy import deepcopy
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
from collections import deque

_mj_envs = {'pen-adroit-v0', 'hammer-adroit-v0', 'door-adroit-v0', 'relocate-adroit-v0',
            'pen-mpl-v0', 'hammer-mpl-v0', 'door-mpl-v0', 'relocate-mpl-v0'}


## DHMEnv 解析配置文件，修改mjcf  按照规范提供接口（dmc_env之类）
## BasicDHMEnv 提供step、reset、obs，不必在意接口

class BasicDHMEnv(gym.Env, ABC):
    def __init__(self, env, feature_type, cfg_dict):
        self.step_count = 0
        self.cfg_dict = cfg_dict
        self._env = env
        self.env_id = env.env.unwrapped.spec.id
        self.random_state = np.random.RandomState(cfg_dict['task']['seed'])
        self.feature_type = feature_type
        self.cam_list = cfg_dict['task']['cam_list']
        self.channels_first = cfg_dict['channels_first']
        self.height = cfg_dict['image_height']
        self.width = cfg_dict['image_width']
        self.action_space = self._env.action_space
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=[3, self.width, self.height], dtype=np.uint8
        )
        self.sim = env.env.sim
        self._env.spec.observation_dim = self.height * self.width * len(self.cam_list)

        if feature_type == 'hybrid':
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
            self.moving_light_step = self.cfg_dict['setting']['moving_light'][self.cfg_dict['task']['moving_light']]['step']
            self.moving_light_range = self.cfg_dict['setting']['moving_light'][self.cfg_dict['task']['moving_light']]['range']
        self.origin_cam_pos = deepcopy(self._env.env.env.model.cam_pos)
        self.origin_cam_fovy = deepcopy(self._env.env.env.model.cam_fovy)
        self.origin_cam_quat = deepcopy(self._env.env.env.model.cam_quat)
        self.origin_light_pos = deepcopy(self._env.env.env.model.light_pos)

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
        qp = None
        if self.feature_type == 'hybrid' or self.feature_type == 'sensor':
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

        pixels = None
        if self.feature_type == 'hybrid' or self.feature_type == 'pixel':
            imgs = []  # number of image is number of camera
            for cam in self.cam_list:  # for each camera, render once
                img = self._env.env.sim.render(width=self.width, height=self.height, mode='offscreen',
                                               camera_name=cam, device_id=0)  # TODO device id will think later
                img = img[::-1, :, :]  # Image given has to be flipped
                if self.channels_first:
                    img = img.transpose((2, 0, 1))  # (3, 224, 224))
                # we should do channels first... (not sure why by default it's not, maybe they did some transpose when using the encoder?)
                # img = img.astype(np.uint8)
                # img = Image.fromarray(img) # TODO is this necessary?
                # plt.imshow(img.transpose((1, 2, 0)))
                assert self.cfg_dict['task']['background']['type'] in ('original', 'color',
                                                                       'video'), 'self.cfg_dict[\'background\'][\'type\'] must in (\'original\', \'color\', \'video\')'
                if self.cfg_dict['task']['background']['type'] == 'video':
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
                imgs.append(img)
            pixels = np.concatenate(imgs, axis=0)

        return pixels, qp

    def get_env_infos(self):
        return self._env.get_env_infos()

    def set_seed(self, seed):
        return self._env.set_seed(seed)

    def reset(self):
        if self.feature_type == 'pixel' or self.feature_type == 'hybrid':
            self.step_count = 0
            color = None
            self.color_index = (self.color_index + 1) % len(self.colors['easy'])
            if self.cfg_dict["task"]["objects_color"] != 'original':
                color = self.colors[self.cfg_dict['task']['objects_color']][
                    self.random_state.randint(len(self.colors[self.cfg_dict['task']['objects_color']]))]['self_rgb']
                # print(self.color_index, np.random.RandomState(self.color_index).randint(len(self.colors[self.cfg_dict['task']['objects_color']])), color)
            # if self.cfg_dict['task']['objects_color'] != 'original':
            #     src_mjcf_path = f'../mj_envs/mj_envs/hand_manipulation_suite/assets/{self.cfg_dict["task"]["taskdef"]["hand"]}/DAPG_assets_template.xml'
            #     dest_mjcf_path = f'../mj_envs/mj_envs/hand_manipulation_suite/assets/{self.cfg_dict["task"]["taskdef"]["hand"]}/DAPG_assets.xml'
            #     color = self.colors[self.cfg_dict['task']['objects_color']][np.random.RandomState(self.color_index).randint(len(self.colors))]['self_rgb']
            #     modify_objects_color(src_mjcf_path, dest_mjcf_path, color)
            #     self._env = GymEnv(self.cfg_dict['task']['env_name'])
            # else:
            #     src_mjcf_path = f'../mj_envs/mj_envs/hand_manipulation_suite/assets/{self.cfg_dict["task"]["taskdef"]["hand"]}/DAPG_assets_template.xml'
            #     dest_mjcf_path = f'../mj_envs/mj_envs/hand_manipulation_suite/assets/{self.cfg_dict["task"]["taskdef"]["hand"]}/DAPG_assets.xml'
            #     modify_objects_color(src_mjcf_path, dest_mjcf_path, None)
            # if self.cfg_dict['task']['table_texture'] != 'original':
            #     src_mjcf_path = f'../mj_envs/mj_envs/hand_manipulation_suite/assets/{self.cfg_dict["task"]["taskdef"]["hand"]}/DAPG_assets.xml'
            #     dest_mjcf_path = f'../mj_envs/mj_envs/hand_manipulation_suite/assets/{self.cfg_dict["task"]["taskdef"]["hand"]}/DAPG_assets.xml'
            #     modify_table_texture(src_mjcf_path, dest_mjcf_path, self.cfg_dict["task"]['table_texture'])
            #     self._env = GymEnv(self.cfg_dict['task']['env_name'])
            src_mjcf_path = f'../mj_envs/mj_envs/hand_manipulation_suite/assets/{self.cfg_dict["task"]["taskdef"]["hand"]}/DAPG_assets_template.xml'
            dest_mjcf_path = f'../mj_envs/mj_envs/hand_manipulation_suite/assets/{self.cfg_dict["task"]["taskdef"]["hand"]}/DAPG_assets.xml'
            modify_assets(src_mjcf_path, dest_mjcf_path, color, self.cfg_dict["task"]['table_texture'],
                          self.color_index % 10)
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
            # if self.cfg_dict['task']['light']['color'] != 'original':
            #     self._env.env.env.model.light_specular[:][0] = self.origin_light_specular[:][0] + self.random_state.uniform(
            #         self.cfg_dict['setting']['light']['color'][self.cfg_dict['task']['light']['color']][0],
            #         self.cfg_dict['setting']['light']['color'][self.cfg_dict['task']['light']['color']][1])
            #     self._env.env.env.model.light_specular[:][1] = self.origin_light_specular[:][
            #                                                       1] + self.random_state.uniform(
            #         self.cfg_dict['setting']['light']['color'][self.cfg_dict['task']['light']['color']][0],
            #         self.cfg_dict['setting']['light']['color'][self.cfg_dict['task']['light']['color']][1])
            #     self._env.env.env.model.light_specular[:][2] = self.origin_light_specular[:][
            #                                                       2] + self.random_state.uniform(
            #         self.cfg_dict['setting']['light']['color'][self.cfg_dict['task']['light']['color']][0],
            #         self.cfg_dict['setting']['light']['color'][self.cfg_dict['task']['light']['color']][1])
            # else:
            #     self._env.env.env.model.light_specular = self.origin_light_specular
            # if self.cfg_dict['task']['light']['intensity'] != 'original':
            #     intensity = self.random_state.uniform(
            #         self.cfg_dict['setting']['light']['intensity'][self.cfg_dict['task']['light']['intensity']][0],
            #         self.cfg_dict['setting']['light']['intensity'][self.cfg_dict['task']['light']['intensity']][1])
            #     self._env.env.env.model.light_specular[:] *= intensity


        self._env.reset()
        pixels, obs_sensor = self.get_obs()
        return pixels, obs_sensor

    def step(self, action):
        if self.cfg_dict['task']['moving_light'] != 'original':
            if abs(self._env.env.env.model.light_pos[:, 1] - self.origin_light_pos[:, 1]) >= self.moving_light_range:
                # print(self._env.env.env.model.light_pos[:,1] - self.origin_light_pos[:,1])
                self.moving_light_step *= -1
            self._env.env.env.model.light_pos[:, 1] += self.moving_light_step
        obs, reward, done, env_info = self._env.step(action)
        obs = self.get_obs()
        return obs, reward, done, env_info

    def set_env_state(self, state):
        return self._env.set_env_state(state)

    def get_env_state(self):
        return self._env.get_env_state()


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
    def __init__(self, cfg_dict):
        cfg_dict['task']['env_name'] = cfg_dict['task']['taskdef']['task'] + '-' + cfg_dict['task']['taskdef'][
            'hand'] + '-' + cfg_dict['task']['taskdef']['version']
        self.env_name = cfg_dict['task']['env_name']
        self.random_state = np.random.RandomState(cfg_dict['task']['seed'])
        # 根据cfg解析修改mjcf DAPG_hammer_template.xml->DAPG_hammer.xml
        src_mjcf_path = f'../mj_envs/mj_envs/hand_manipulation_suite/assets/{cfg_dict["task"]["taskdef"]["hand"]}/DAPG_{cfg_dict["task"]["taskdef"]["task"]}_template.xml'
        dest_mjcf_path = f'../mj_envs/mj_envs/hand_manipulation_suite/assets/{cfg_dict["task"]["taskdef"]["hand"]}/DAPG_{cfg_dict["task"]["taskdef"]["task"]}_test.xml'
        modify_mjcf_via_dict(src_mjcf_path, dest_mjcf_path, cfg_dict['task'], self.random_state, cfg_dict['setting'])

        default_env_to_cam_list = {
            'hammer-adroit-v0': ['side'],
            'door-adroit-v0': ['top'],
            'pen-adroit-v0': ['vil_camera'],
            'relocate-adroit-v0': ['cam1', 'cam2', 'cam3'],
        }
        if cfg_dict['task']['cam_list'] is None:
            cfg_dict['task']['cam_list'] = default_env_to_cam_list[self.env_name]

        env = GymEnv(self.env_name)

        env = BasicDHMEnv(env, feature_type=cfg_dict['feature_type'], cfg_dict=cfg_dict)

        self._env = env
        self.obs_dim = env.spec.observation_dim
        self.obs_sensor_dim = 24
        self.act_dim = env.spec.action_dim
        self.horizon = env.spec.horizon
        number_channel = len(cfg_dict['task']['cam_list']) * 3

        if cfg_dict['feature_type'] == 'hybrid':
            self._obs_spec = specs.BoundedArray(
                shape=(number_channel, cfg_dict['image_height'], cfg_dict['image_width']), dtype='uint8',
                name='observation', minimum=0, maximum=255)
            self._obs_sensor_spec = specs.Array(shape=(self.obs_sensor_dim,), dtype='float32',
                                                name='observation_sensor')
        elif cfg_dict['feature_type'] == 'pixel':
            self._obs_spec = specs.BoundedArray(
                shape=(number_channel, cfg_dict['image_height'], cfg_dict['image_width']), dtype='uint8',
                name='observation', minimum=0, maximum=255)
        elif cfg_dict['feature_type'] == 'sensor':
            self._obs_sensor_spec = specs.Array(shape=(self.obs_sensor_dim,), dtype='float32',
                                                name='observation_sensor')
        self._action_spec = specs.BoundedArray(shape=(self.act_dim,), dtype='float32', name='action', minimum=-1.0,
                                               maximum=1.0)

    def reset(self):
        # pixels and sensor values
        obs_pixels, obs_sensor = self._env.reset()
        obs_sensor = obs_sensor.astype(np.float32)
        action_spec = self.action_spec()
        action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        time_step = ExtendedTimeStep(observation=obs_pixels,
                                     observation_sensor=obs_sensor,
                                     step_type=StepType.FIRST,
                                     action=action,
                                     reward=0.0,
                                     discount=1.0, )
        return time_step

    def get_current_obs_without_reset(self):
        # use this to obtain the first state in a demo
        obs_pixels, obs_sensor = self._env.get_obs_for_first_state_but_without_reset()
        obs_sensor = obs_sensor.astype(np.float32)
        action_spec = self.action_spec()
        action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        time_step = ExtendedTimeStep(observation=obs_pixels,
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

        # currently it seems there is simply no actual terminal state, so let's not worry about the discount..
        # TODO but might be good to test in drq dmc code to make sure we know what's going on
        discount = 1.0

        if done:
            steptype = StepType.LAST
        else:
            steptype = StepType.MID

        if force_step_type is not None:
            if force_step_type == 'mid':
                steptype = StepType.MID
            elif force_step_type == 'last':
                steptype = StepType.LAST
            else:
                steptype = StepType.FIRST

        time_step = ExtendedTimeStep(observation=obs_pixels,
                                     observation_sensor=obs_sensor,
                                     step_type=steptype,
                                     action=action,
                                     reward=reward,
                                     discount=discount, )
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

    # def __getattr__(self, name):
    #     return getattr(self, name)


import hydra
def make_env(task):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    origin_dir = os.getcwd()
    current_dir = os.path.dirname(__file__)
    os.chdir(current_dir)
    with initialize(config_path="../cfg"):
        cfg = compose(config_name="config", overrides=[f"task={task}"])
        cfg_dict = omegaconf_to_dict(cfg)
    env = DHMEnv(cfg_dict)
    os.chdir(origin_dir)
    return env


if __name__ == '__main__':
    env = make_env('hammer')
