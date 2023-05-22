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
from wrappers.robo_wrapper import robo_make
import imageio
from tqdm import tqdm
import cv2
from algos import pieg

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        if self.cfg.env == 'habitat':
            self.appearance_id = cfg.wandb_group.split('_')[2]
        
        self.setup()

        self.level = cfg.wandb_group.split('_')[0]
        assert self.level in ['train', 'test']
        self.agent_name = cfg.wandb_group.split('_')[1]
        work_dir = f'/home/yzc/shared/project/mujoco_manipulation/drqv2/exp_local/robo_model/{self.cfg.task_name}/{self.agent_name}/{cfg.seed}'
        self.model_work_dir = work_dir
        self.agent = torch.load('%s/snapshot.pt' % (work_dir), map_location='cuda:0')['agent']
        self.timer = utils.Timer()
        # if self.cfg.task_name == 'Door':
        #     self._global_step = int(3e5)
        # elif self.cfg.task_name == 'TwoArmPegInHole':
        #     self._global_step = int(4e5)
        # elif self.cfg.task_name == 'Lift':
        #     self._global_step = int(4e5)
        if self.cfg.env == 'robosuite':
            assert self.cfg.task_name in ['Door', 'TwoArmPegInHole', 'Lift']
        elif self.cfg.env == 'habitat':
            assert self.cfg.task_name == 'habitat'

        self._global_step = agent['_global_step']
        self._global_episode = 0

    def setup(self):
        if self.cfg.use_wandb:
            exp_name = '_'.join([
                self.cfg.task_name,
                str(self.cfg.seed)
            ])
            wandb.init(project="gvrlb_algo", group=self.cfg.wandb_group, name=exp_name)
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb, use_wandb=self.cfg.use_wandb)
        # create envs
        if self.cfg.env == 'robosuite':
            from wrappers.robo_wrapper import robo_make
            self.train_env = robo_make(name=self.cfg.task_name, action_repeat=self.cfg.action_repeat, frame_stack=self.cfg.frame_stack, seed=self.cfg.seed)
            self.eval_env = robo_make(name=self.cfg.task_name, action_repeat=self.cfg.action_repeat, frame_stack=self.cfg.frame_stack, seed=self.cfg.seed)
        elif self.cfg.env == 'habitat':
            hydra.core.global_hydra.GlobalHydra.instance().clear()
            from wrappers.habi_wrapper import make_habitat_env
            self.eval_env = make_habitat_env(name='HabitatImageNav-v0', mode='test', seed=self.cfg.seed,
                                    action_repeat=self.cfg.action_repeat, appearance_id=self.appearance_id)
        else:
            raise ValueError(f"env {env} not supported.")
            

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)


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
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        count = 0
        for i in tqdm(range(1, 101)):
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
            self.video_recorder.save(f'{self.global_frame}.mp4')
        print(f'Seed {self.cfg.seed} Mean_reward: ', total_reward / episode)

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    
    
    
    def habi_eval(self):
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
    if cfg.env == 'robo':
        workspace.robo_eval()
    elif cfg.env == 'habitat':
        workspace.habi_eval()
    else:
        workspace.eval()


if __name__ == '__main__':
    main()