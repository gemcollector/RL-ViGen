# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import wrappers.dmc as dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import wandb
import imageio
from tqdm import tqdm
import cv2
import sys
import matplotlib.pyplot as plt
sys.path.append('./algos')
from wrappers.carlaenv.utils import make_env_10, make_env_10_eval

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        """
        set up environments and agents.
        """
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.seed = cfg.seed
        self.device = torch.device(cfg.device)
        if self.cfg.env == 'habitat':
            self.appearance_id = cfg.wandb_group.split('_')[2]
        
        self.setup()

        self.level = cfg.wandb_group.split('_')[0]
        assert self.level in ['train', 'test']
        self.agent_name = cfg.wandb_group.split('_')[1]
        work_dir = f'{cfg.model_dir}/{self.agent_name}/{cfg.seed}'
        self.model_work_dir = work_dir
        agent = torch.load('%s/snapshot.pt' % (work_dir), map_location='cuda:0')
        self.agent = agent['agent']
        self.timer = utils.Timer()
        assert self.cfg.env in ['carla', 'robosuite', 'habitat']
        if self.cfg.env == 'robosuite':
            assert self.cfg.task_name in ['Door', 'TwoArmPegInHole', 'Lift']
        elif self.cfg.env == 'habitat':
            assert self.cfg.task_name == 'habitat'
        elif self.cfg.env == 'carla':
            assert self.cfg.task_name == 'carla'

        self._global_step = agent['_global_step']
        print('global_step: ', self._global_step)
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb, use_wandb=self.cfg.use_wandb)
        # create envs
        if self.cfg.env == 'robosuite':
            from wrappers.robo_wrapper import robo_make
            self.train_env = robo_make(name=self.cfg.task_name, action_repeat=self.cfg.action_repeat, frame_stack=self.cfg.frame_stack, seed=self.cfg.seed)
            self.eval_env = robo_make(name=self.cfg.task_name, action_repeat=self.cfg.action_repeat, frame_stack=self.cfg.frame_stack, seed=self.cfg.seed)
        elif self.cfg.env == 'habitat':
            os.environ["GLOG_minloglevel"] = "3"
            os.environ["MAGNUM_LOG"] = "quiet"
            hydra.core.global_hydra.GlobalHydra.instance().clear()
            from wrappers.habi_wrapper import make_habitat_env
            self.eval_env = make_habitat_env(name='HabitatImageNav-v0', mode='test', seed=self.cfg.seed,
                                    action_repeat=self.cfg.action_repeat, appearance_id=self.appearance_id)
        elif self.cfg.env == 'carla':
            from wrappers.carla_wrapper import carla_make_eval
            self.eval_env = carla_make_eval(action_repeat=self.cfg.action_repeat)
        else:
            raise ValueError(f"env {self.cfg.env} not supported.")
            

        self.video_recorder = None


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat


    def robo_record_gif(self):
        total_reward = 0
        images = []
        timestep = self.eval_env.reset()
        obs = self.eval_env._gym_env.env.sim.render(
            mode="offscreen",
            width=224,
            height=224,
            camera_name='agentview',
        )
        images.append(obs[::-1, :, :])
        while not timestep.last():
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(timestep.observation, self.global_step, eval_mode=True)
            timestep = self.eval_env.step(action)
            obs = self.eval_env._gym_env.env.sim.render(
                mode="offscreen",
                width=224,
                height=224,
                camera_name='agentview')
            images.append(obs[::-1, :, :])
            total_reward += timestep.reward

        print('total_reward: ', total_reward)

        imageio.mimsave('%s/%s.gif' % (self.work_dir, self.agent_name), [np.array(img) for i, img in enumerate(images) if i % 1 == 0], fps=15)




    def robo_eval(self):
        """evaluate on robosuite."""
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        count = 0
        for i in tqdm(range(1, 101)):
            episode_reward = 0
            time_step = self.eval_env.reset()
            while not time_step.last():
                if self.agent_name == 'pieg':
                    with torch.no_grad():
                        action = self.agent.act(time_step.observation,
                                                self.global_step,
                                                eval_mode=True)
                else:
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(time_step.observation,
                                                self.global_step,
                                                eval_mode=True)
                time_step = self.eval_env.step(action)
                total_reward += time_step.reward
                episode_reward += time_step.reward
                step += 1
            f = open("{}/file_{}.txt".format(self.model_work_dir, self.cfg.seed), 'a')
            f.write("episode_reward: %f \n" % (float(episode_reward)))
            f.close()

            if self.level == 'train':
                pass
            else:
                if i < 100 and i % 10 == 0:
                    count += 1
                    print(f'==switch to the new scene {count}_id==')
                    self.eval_env = robo_make(name=self.cfg.task_name, action_repeat=self.cfg.action_repeat, frame_stack=self.cfg.frame_stack, seed=self.cfg.seed, scene_id=count)

            episode += 1
        print(f'Seed {self.cfg.seed} Mean_reward: ', total_reward / episode)

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    
    
    
    def habi_eval(self):
        """evaluate on habitat."""
        print(f'test_{self.agent_name}_{self.appearance_id}')
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        count = 0
        success_rate = 0
        for i in tqdm(range(1, 11)):
            episode_reward = 0
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                if self.agent_name == 'pieg':
                    with torch.no_grad():
                        action = self.agent.act(time_step.observation,
                                                self.global_step,
                                                eval_mode=True)
                else:
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(time_step.observation,
                                                self.global_step,
                                                eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                episode_reward += time_step.reward
                step += 1

            success_rate += time_step.info['success']
            f = open("{}/file_{}.txt".format(self.model_work_dir, self.cfg.seed), 'a')
            f.write("episode_reward: %f \n" % (float(episode_reward)))
            f.write("success_rate: %f \n" % (float(time_step.info['success'])))
            f.close()

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')
        print(f'Seed {self.cfg.seed} Mean_reward: ', total_reward / episode)
        print(f'Seed {self.cfg.seed} Success_rate: ', success_rate / episode)

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            log('success_rate', success_rate / episode)
    
    
    
    def carla_eval(self):
        """evaluate on carla."""
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        # carla metrics:
        reason_each_episode_ended = []
        distance_driven_each_episode = []
        crash_intensity = 0.
        steer = 0.
        brake = 0.
        count = 0
        success_num = 0

        for i in range(50):
            time_step = self.eval_env.reset()
            # To check wether the weather is successfully changed
            if i == 0:
                plt.imshow(time_step.observation[6:9].transpose(1, 2, 0) / 255.)
                plt.savefig(f'{self.work_dir}/test.png')
            # self.video_recorder.init(enabled=True)
            dist_driven_this_episode = 0.
            while not time_step.last():
                if self.agent_name == 'pieg':
                    with torch.no_grad():
                        action = self.agent.act(time_step.observation,
                                                self.global_step,
                                                eval_mode=True)
                else:
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(time_step.observation,
                                                self.global_step,
                                                eval_mode=True)
                time_step, info = self.eval_env.step(action)
                # self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

                dist_driven_this_episode += info['distance']
                crash_intensity += info['crash_intensity']
                steer += abs(info['steer'])
                brake += info['brake']
                count += 1

            episode += 1
            print('total_reward per episode:', total_reward / episode)
            # self.video_recorder.save(f'{episode}.mp4')

            reason_each_episode_ended.append(info['reason_episode_ended'])
            distance_driven_each_episode.append(dist_driven_this_episode)
            if info['reason_episode_ended'] == 'success':
                success_num += 1


        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            log('success_rate', success_num / episode)

        print('METRICS--------------------------')
        print("reason_each_episode_ended: {}".format(reason_each_episode_ended))
        print("distance_driven_each_episode: {}".format(distance_driven_each_episode))
        print('crash_intensity: {}'.format(crash_intensity / self.cfg.num_eval_episodes))
        print('steer: {}'.format(steer / count))
        print('brake: {}'.format(brake / count))
        print('---------------------------------')
        f = open("{}/file_{}.txt".format(self.work_dir, self.seed), 'a')
        f.write("seed: %f \n" % (self.seed))
        f.write("weather_name: %s \n" % (self.env_weather_name))
        f.write("reward: %f \n" % (float(total_reward / episode)))
        f.write("distance: %f \n" % (float(np.mean(distance_driven_each_episode))))
        f.write("steer: %f \n" % (steer / count))
        f.write("brake: %f \n" % (brake / count))
        f.write("reason_episode_ended: {} \n".format(reason_each_episode_ended))
        f.write("distance all: {} \n".format(distance_driven_each_episode))
        f.write("=========================================== \n")
        f.close()
    
    
    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from eval import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    # workspace.record_gif()
    if cfg.env == 'robosuite':
        workspace.robo_eval()
    elif cfg.env == 'habitat':
        workspace.habi_eval()
    elif cfg.env == 'carla':
        workspace.carla_eval()
        workspace.eval_env.finish()
    else:
        workspace.eval()


if __name__ == '__main__':
    main()