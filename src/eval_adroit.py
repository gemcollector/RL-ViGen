# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import dhm
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from distutils.dir_util import copy_tree
import torchvision.transforms as transforms
from pathlib import Path
import cv2
import imageio
import joblib

import hydra
import torch
from dm_env import StepType, TimeStep, specs

import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import pickle
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True

# TODO this part can be done during workspace setup
ENV_TYPE = 'adroit'
if ENV_TYPE == 'adroit':
    from adroit import AdroitEnv
else:
    import dmc
IS_ADROIT = True if ENV_TYPE == 'adroit' else False



def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.direct_folder_name = os.path.basename(self.work_dir)

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.replay_buffer_fetch_every = 1000
        assert len(cfg.wandb_group.split('_')) == 3
        self.level = cfg.wandb_group.split('_')[2]
        if self.cfg.debug > 0:  # if debug mode, then change hyperparameters for quick testing
            self.set_debug_hyperparameters()
        self.setup()

        print(cfg.wandb_group)
        self.agent_name = cfg.wandb_group.split('_')[1]
        self.cfg.task_name = self.cfg.task_name.split('-')[0]
        work_dir = cfg.model_dir
        self.model_work_dir = work_dir
        num = 2000000 if self.cfg.task_name == 'pen' else 1000000
        device_id = self.cfg.device.split(':')[-1]
        self.agent = torch.load(f'%s/snapshot{num}.pt' % (work_dir), map_location=f'cuda:{device_id}')['agent']
        self.agent.device = self.cfg.device
        self.agent.encoder = self.agent.encoder.to(self.cfg.device)
        self.agent.actor = self.agent.actor.to(self.cfg.device)
        self.agent.critic = self.agent.critic.to(self.cfg.device)
        self.timer = utils.Timer()

        if self.cfg.task_name == 'pen':
            self._global_step = int(1e6)
        else:
            self._global_step = int(5e5)
        self._global_episode = 0

    def set_debug_hyperparameters(self):
        self.cfg.num_seed_frames = 1000 if self.cfg.num_seed_frames > 1000 else self.cfg.num_seed_frames
        self.cfg.agent.num_expl_steps = 500 if self.cfg.agent.num_expl_steps > 1000 else self.cfg.agent.num_expl_steps
        if self.cfg.replay_buffer_num_workers > 1:
            self.cfg.replay_buffer_num_workers = 1
        self.cfg.num_eval_episodes = 1
        self.cfg.replay_buffer_size = 30000
        self.cfg.batch_size = 8
        self.cfg.feature_dim = 8
        self.cfg.num_train_frames = 5050
        self.replay_buffer_fetch_every = 30
        self.cfg.stage2_n_update = 100
        self.cfg.num_demo = 3
        self.cfg.eval_every_frames = 3000
        self.cfg.agent.hidden_dim = 8
        self.cfg.agent.num_expl_steps = 500
        self.cfg.stage2_eval_every_frames = 50

    def create_adroit_env(self):

        self.train_env = dhm.make_env_RRL(self.env_name, test_image=False, num_repeats=self.cfg.action_repeat,
                                          num_frames=self.cfg.frame_stack, env_feature_type=self.env_feature_type,
                                          device=self.device, reward_rescale=self.cfg.reward_rescale, mode='test', seed=self.cfg.seed, level=self.level)
        self.eval_env = dhm.make_env_RRL(self.env_name, test_image=False, num_repeats=self.cfg.action_repeat,
                                         num_frames=self.cfg.frame_stack, env_feature_type=self.env_feature_type,
                                         device=self.device, reward_rescale=self.cfg.reward_rescale, mode='test', seed=self.cfg.seed, level=self.level)

    def setup(self):
        warnings.filterwarnings('ignore', category=DeprecationWarning)

        if self.cfg.save_models:
            assert self.cfg.action_repeat % 2 == 0

        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        env_name = self.cfg.task_name
        self.env_name = env_name
        env_type = 'adroit' if env_name in ('hammer-v0', 'door-v0', 'pen-v0', 'relocate-v0') else 'dmc'
        # assert env_name in ('hammer-v0','door-v0','pen-v0','relocate-v0',)

        if self.cfg.agent.encoder_lr_scale == 'auto':
            if env_name == 'relocate-v0':
                self.cfg.agent.encoder_lr_scale = 0.01
            else:
                self.cfg.agent.encoder_lr_scale = 1

        self.env_feature_type = self.cfg.env_feature_type
        if env_type == 'adroit':
            self.train_env = dhm.make_env_RRL(env_name, test_image=False, num_repeats=self.cfg.action_repeat,
                                              num_frames=self.cfg.frame_stack, env_feature_type=self.env_feature_type,
                                              device=self.device, reward_rescale=self.cfg.reward_rescale, mode='test', seed=self.cfg.seed, level=self.level)
            self.eval_env = dhm.make_env_RRL(env_name, test_image=False, num_repeats=self.cfg.action_repeat,
                                             num_frames=self.cfg.frame_stack, env_feature_type=self.env_feature_type,
                                             device=self.device, reward_rescale=self.cfg.reward_rescale, mode='test', seed=self.cfg.seed, level=self.level)

            data_specs = (self.train_env.observation_spec(),
                          self.train_env.observation_sensor_spec(),
                          self.train_env.action_spec(),
                          specs.Array((1,), np.float32, 'reward'),
                          specs.Array((1,), np.float32, 'discount'),
                          specs.Array((1,), np.int8, 'n_goal_achieved'),
                          specs.Array((1,), np.float32, 'time_limit_reached'),
                          )

        # create replay buffer
        self.replay_storage = ReplayBufferStorage(data_specs, self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount, self.replay_buffer_fetch_every,
            is_adroit=IS_ADROIT)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)

    def set_demo_buffer_nstep(self, nstep):
        self.replay_loader_demo.dataset._nstep = nstep

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

    def eval_adroit(self, force_number_episodes=None, do_log=True):

        step, episode, total_reward = 0, 0, 0
        n_eval_episode = 100
        total_success = 0.0
        for i in tqdm(range(n_eval_episode)):
            n_goal_achieved_total = 0
            time_step = self.eval_env.reset()
            # plt.imshow(time_step.observation[6:9].transpose(1, 2, 0))
            # plt.savefig('./scene.png')
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                if self.agent_name == 'pieg':
                    with torch.no_grad():
                        observation = time_step.observation
                        action = self.agent.act(observation,
                                                self.global_step,
                                                eval_mode=True,
                                                obs_sensor=time_step.observation_sensor)
                else:
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        observation = time_step.observation
                        action = self.agent.act(observation,
                                                self.global_step,
                                                eval_mode=True,
                                                obs_sensor=time_step.observation_sensor)
                time_step = self.eval_env.step(action)
                n_goal_achieved_total += time_step.n_goal_achieved
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            # here check if success for Adroit tasks. The threshold values come from the mj_envs code
            # e.g. https://github.com/ShahRutav/mj_envs/blob/5ee75c6e294dda47983eb4c60b6dd8f23a3f9aec/mj_envs/hand_manipulation_suite/pen_v0.py
            # can also use the evaluate_success function from Adroit envs, but can be more complicated
            if self.cfg.task_name == 'pen-v0':
                threshold = 20
            else:
                threshold = 25
            if n_goal_achieved_total > threshold:
                total_success += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')
        success_rate_standard = total_success / n_eval_episode
        episode_reward_standard = total_reward / episode
        episode_length_standard = step * self.cfg.action_repeat / episode

        f = open("{}/file_{}.txt".format(self.model_work_dir, self.cfg.seed), 'a')
        f.write("episode_reward: %f \n" % (float(episode_reward_standard)))
        f.write("success_rate: %f \n" % (float(success_rate_standard)))
        f.close()
        print(f"Seed {self.cfg.seed}  ||  Success rate: {success_rate_standard}")

        if do_log:
            with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
                log('episode_reward', episode_reward_standard)
                log('success_rate', success_rate_standard)
                log('episode_length', episode_length_standard)
                log('episode', self.global_episode)
                log('step', self.global_step)

        return episode_reward_standard, success_rate_standard


    def save_snapshot(self, suffix=None):
        if suffix is None:
            save_name = 'snapshot.pt'
        else:
            save_name = 'snapshot' + suffix + '.pt'
        snapshot = self.work_dir / save_name
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)
        print("snapshot saved to:", str(snapshot))

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v



        step, episode, total_reward = 0, 0, 0
        images = []
        camera_name = 'vil_camera'
        timestep = self.eval_env.reset()

        n_eval_episode = 1
        total_success = 0.0
        for i in tqdm(range(n_eval_episode)):
            n_goal_achieved_total = 0
            time_step = self.eval_env.reset()
            obs = self.eval_env._env._env.env.sim.render(width=224, height=224, mode='offscreen', camera_name=camera_name, device_id=0)
            if camera_name == 'vil_camera':
                obs = obs[::-1, :, :]
            images.append(obs)

            # images.append(time_step.observation[6:9].transpose(1, 2, 0))
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    observation = time_step.observation
                    action = self.agent.act(observation,
                                            self.global_step,
                                            eval_mode=True,
                                            obs_sensor=time_step.observation_sensor)
                time_step = self.eval_env.step(action)
                obs = self.eval_env._env._env.env.sim.render(width=224, height=224, mode='offscreen', camera_name=camera_name,
                                                             device_id=0)
                if camera_name == 'vil_camera':
                    obs = obs[::-1, :, :]
                images.append(obs)
                # images.append(time_step.observation[6:9].transpose(1, 2, 0))
                n_goal_achieved_total += time_step.n_goal_achieved
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            # here check if success for Adroit tasks. The threshold values come from the mj_envs code
            # e.g. https://github.com/ShahRutav/mj_envs/blob/5ee75c6e294dda47983eb4c60b6dd8f23a3f9aec/mj_envs/hand_manipulation_suite/pen_v0.py
            # can also use the evaluate_success function from Adroit envs, but can be more complicated
            if self.cfg.task_name == 'pen-v0':
                threshold = 20
            else:
                threshold = 25
            if n_goal_achieved_total > threshold:
                total_success += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')
        success_rate_standard = total_success / n_eval_episode
        episode_reward_standard = total_reward / episode
        episode_length_standard = step * self.cfg.action_repeat / episode

        print(f"Success rate: {success_rate_standard}")
        print(f"Episode reward: {episode_reward_standard}")

        imageio.mimsave('%s/%s.gif' % (self.work_dir, self.agent_name), [np.array(img) for i, img in enumerate(images) if i % 1 == 0], fps=15)


    def record_video_gif(self):
        """
        visualize your own trained model in vil_camera view.
        """

        step, episode, total_reward = 0, 0, 0
        images = []
        camera_name = 'vil_camera'
        # timestep = self.eval_env.reset()
        n_eval_episode = 1
        total_success = 0.0
        for i in tqdm(range(n_eval_episode)):
            count = 0
            n_goal_achieved_total = 0
            time_step = self.eval_env.reset()
            image = self.eval_env.render(image_size=256, camera_name=camera_name)
            images.append(image[::-1, :, :])
            # images.append(obs)
            # images.append(time_step.observation[6:9].transpose(1, 2, 0)[::-1, :, :])
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    observation = time_step.observation
                    action = self.agent.act(observation,
                                            self.global_step,
                                            eval_mode=True,
                                            obs_sensor=time_step.observation_sensor)
                time_step = self.eval_env.step(action)
                image = self.eval_env.render(image_size=256, camera_name=camera_name)
                images.append(image[::-1, :, :])
                # images.append(time_step.observation[6:9].transpose(1, 2, 0)[::-1, :, :])
                n_goal_achieved_total += time_step.n_goal_achieved
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1
                count += 1
                if count == 40:
                    break

            # here check if success for Adroit tasks. The threshold values come from the mj_envs code
            # e.g. https://github.com/ShahRutav/mj_envs/blob/5ee75c6e294dda47983eb4c60b6dd8f23a3f9aec/mj_envs/hand_manipulation_suite/pen_v0.py
            # can also use the evaluate_success function from Adroit envs, but can be more complicated
            if self.cfg.task_name == 'pen-v0':
                threshold = 20
            else:
                threshold = 25
            if n_goal_achieved_total > threshold:
                total_success += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')
        success_rate_standard = total_success / n_eval_episode
        episode_reward_standard = total_reward / episode
        episode_length_standard = step * self.cfg.action_repeat / episode

        print(f"Success rate: {success_rate_standard}")
        print(f"Episode reward: {episode_reward_standard}")

        imageio.mimsave('%s/%s.gif' % (self.work_dir, self.agent_name), [np.array(img) for i, img in enumerate(images) if i % 1 == 0], fps=15)





    def del_xml(self):
        self.train_env.del_xml()



@hydra.main(config_path='cfgs_adroit', config_name='config')
def main(cfg):
    # TODO potentially check the task name and decide which libs to load here?
    W = Workspace
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    # workspace.record_video_gif()
    workspace.eval_adroit()
    workspace.del_xml()


if __name__ == '__main__':
    main()