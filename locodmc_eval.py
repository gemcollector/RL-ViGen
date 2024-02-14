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

import wrappers.loco_wrapper as dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import wandb
from tqdm import tqdm
import sys
sys.path.append('./algos')

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
        self.setup()
        self.agent_name = cfg.wandb_group.split('_')[1]
        print(f'agent name: {self.agent_name}')
        work_dir = f'{cfg.model_dir}/{self.agent_name}/{cfg.seed}'
        self.model_work_dir = work_dir
        agent = torch.load('%s/snapshot.pt' % (work_dir), map_location='cuda:0')
        self.agent = agent['agent']
        self._global_step = agent['_global_step']


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
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed, type='video', difficulty='hard')
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed, type='video', difficulty='hard')
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = None
        self.train_video_recorder = None


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        num_episodes = 100
        for i in tqdm(range(num_episodes)):
            time_step = self.eval_env.reset()
            self.video_recorder.init_dmc(self.eval_env, enabled=True)
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
                self.video_recorder.record_dmc(self.eval_env, video=True)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{i}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            # log('episode', self.global_episode)
            log('step', self.global_step)

        episode_reward_standard = total_reward / episode
        print(f"Episode reward: {episode_reward_standard}")
        
        


    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='curl_config')
def main(cfg):
    from locodmc_eval import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()

    workspace.eval()


if __name__ == '__main__':
    main()