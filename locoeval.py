import torch
import torchvision
import os
import numpy as np
import gym
import utils
from copy import deepcopy
from tqdm import tqdm
from dmcvgb.make_env import make_env
import wrappers.dmc as dmc
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path
from dm_control.suite.wrappers import action_scale
from video import VideoRecorder
import imageio
import sys
import matplotlib.pyplot as plt
sys.path.append('./algos')


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--domain_name', default='walker')
parser.add_argument('--task_name', default='walk')
parser.add_argument('--algorithm', default=None, type=str)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--eval_episodes', default=100, type=int)
parser.add_argument('--model_dir', default=None, type=str)
parser.add_argument('--save_video', default=False, action='store_true')
parser.add_argument('--mode', default='train', type=str)



class DMCVideoRecoder(VideoRecorder):
    def __init__(self, root_dir, camera_id, render_size=256, fps=20):
        super().__init__(root_dir, render_size, fps)
        self.height = render_size
        self.width = render_size
        self.camera_id = camera_id

    def init(self, env, mode, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env, mode)


    def record(self, env, mode):
        if self.enabled:
            frame = env.render(
                mode='rgb_array',
                height=self.height,
                width=self.width,
                camera_id=self.camera_id
            )
            if mode is not None and 'video' in mode:
                _env = env
                while 'video' not in _env.__class__.__name__.lower():
                    _env = _env.env
                frame = _env.apply_to(frame)
            self.frames.append(frame)


def evaluate(env, agent, video, video_dir, mode, num_episodes, seed=0, domain_name='cartpole', task_name='swingup', algo='svea', step=int(5e5)):
    episode_rewards = []
    count = 0
    for i in tqdm(range(num_episodes)):
        ep_agent = agent
        try:
            obs = env.reset()
        except:
            obs = env.reset()
        video.init(env, mode, enabled=True)
        done = False
        episode_reward = 0
        while not done:
            count += 1
            if algo == 'pieg':
                with torch.no_grad():
                    action = ep_agent['agent'].act(np.array(obs),
                                                   step,
                                                   eval_mode=True)
            else:
                with torch.no_grad(), utils.eval_mode(ep_agent['agent']):
                    action = ep_agent['agent'].act(np.array(obs),
                                                step,
                                                eval_mode=True)

            next_obs, reward, done, _ = env.step(action)
            video.record(env, mode)
            episode_reward += reward

            obs = next_obs

        video.save(f'{video_dir}/eval_{i}.mp4')
        episode_reward = 0 if episode_reward < 0 else episode_reward
        episode_rewards.append(episode_reward)


    return np.mean(episode_rewards)


def main(args):
    # Set seed
    # Initialize environments
    # gym.logger.set_level(40)
    domain_name = args.domain_name
    task_name = args.task_name
    print(f'task name: {domain_name} {task_name}')
    algorithm = args.algorithm
    utils.set_seed_everywhere(args.seed)
    env = make_env(domain_name, task_name, args.seed)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

    work_dir = Path.cwd()
    print(f'workspace: {work_dir}')
    if domain_name == 'unitree' or domain_name == 'quadruped':
        camera_id = 1
    else:
        camera_id = 0



    model_dir = args.model_dir + f'{domain_name}_{task_name}/{algorithm}/{args.seed}'
    agent = torch.load('%s/snapshot.pt' % (model_dir), map_location='cuda:0')
    step = agent['_global_step']
    agent['agent'].device = torch.device('cuda:0')
    if algorithm != 'pieg':
        agent['agent'].train(False)
    eval_episodes = args.eval_episodes if not args.save_video else 10
    video_dir = utils.make_dir(os.path.join(model_dir, 'video'))
    video = DMCVideoRecoder(Path(video_dir) if args.save_video else None, camera_id=camera_id, render_size=448)

    reward = evaluate(env, agent, video, video_dir, args.mode, num_episodes=eval_episodes, seed=args.seed, domain_name=domain_name, task_name=task_name, algo=algorithm, step=step)
    print(f'Seed {args.seed},  Reward:', int(reward))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)