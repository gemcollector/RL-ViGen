from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np

SUITE = containers.TaggedTasks()

_CONTROL_TIMESTEP = .01  # (Seconds)
# Default duration of an episode, in seconds.
_DEFAULT_TIME_LIMIT = 10  # TODO


def get_model_and_assets_reach():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model(
        'mujoco_menagerie/franka_emika_panda/scene_reach.xml'), common.ASSETS

def get_model_and_assets_push():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model(
        'mujoco_menagerie/franka_emika_panda/scene_push.xml'), common.ASSETS


@SUITE.add('benchmarking')
def reach(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns a Franka that strives to stand upright, balancing its pose."""
    origin_dir = os.getcwd()
    current_dir = os.path.dirname(__file__)
    os.chdir(current_dir + '/mujoco_menagerie/franka_emika_panda')
    physics = Physics.from_xml_string(*get_model_and_assets_reach())
    os.chdir(origin_dir)
    task = Franka(task='reach')
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)

@SUITE.add('benchmarking')
def push(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns a Franka that strives to stand upright, balancing its pose."""
  origin_dir = os.getcwd()
  current_dir = os.path.dirname(__file__)
  os.chdir(current_dir + '/mujoco_menagerie/franka_emika_panda')
  physics = Physics.from_xml_string(*get_model_and_assets_push())
  os.chdir(origin_dir)
  task = Franka(task='push')
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


class Physics(mujoco.Physics):
    def end_to_target(self):
        data = self.named.data
        end_to_target = data.site_xpos['target_ball'] - (data.xpos['right_finger'] + data.xpos['left_finger']) / 2
        return np.linalg.norm(end_to_target)

    def lfinger_to_target(self):
        data = self.named.data
        end_to_target = data.site_xpos['target_ball'] - data.site_xpos['left_finger']
        return np.linalg.norm(end_to_target)

    def rfinger_to_target(self):
        data = self.named.data
        end_to_target = data.site_xpos['target_ball'] - data.site_xpos['right_finger']
        return np.linalg.norm(end_to_target)

    def object_to_otarget(self):
        data = self.named.data
        end_to_target = data.site_xpos['target_ball'] - data.site_xpos['target_box']
        return np.linalg.norm(end_to_target)


class Franka(base.Task):
    def __init__(self, task, random=None):
        super(Franka, self).__init__(random=random)
        self.task = task

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        physics.reset()
        if self.task == 'reach':
            physics.named.model.site_pos['target_ball', 0] = np.random.uniform(low=0.4, high=0.6)
            physics.named.model.site_pos['target_ball', 1] = np.random.uniform(low=-0.1, high=0.1)
            physics.named.model.site_pos['target_ball', 2] = np.random.uniform(low=0.65, high=0.85)
        elif self.task == 'push':
            physics.named.model.site_pos['target_box', 0] = np.random.uniform(low=0.7, high=0.85)
            physics.named.model.site_pos['target_box', 1] = np.random.uniform(low=-0.1, high=0.1)
            physics.named.model.body_pos['object_box', 0] = np.random.uniform(low=0.43, high=0.58)
            physics.named.model.body_pos['object_box', 1] = np.random.uniform(low=-0.1, high=0.1)
        super(Franka, self).initialize_episode(physics)

    def get_observation(self, physics):
        # TODO
        obs = collections.OrderedDict()
        obs['position'] = physics.data.qpos[:].copy()
        obs['velocity'] = physics.data.qvel[:].copy()
        obs['right_finger'] = physics.named.data.site_xpos['right_finger'].copy()
        obs['left_finger'] = physics.named.data.site_xpos['left_finger'].copy()
        obs['target'] = physics.named.data.site_xpos['target_ball'].copy()
        if self.task == 'push':
            obs['target_box'] = physics.named.data.site_xpos['target_box'].copy()
        return obs

    def before_step(self, action, physics):
        action = getattr(action, "continuous_actions", action)
        self.current_action = action
        physics.set_control(action)

    def get_reward(self, physics):
        minimum = self.action_spec(physics).minimum
        maximum = self.action_spec(physics).maximum
        orig_minimum = np.array([-1.0 for i in range(self.current_action.shape[0])])
        scale = (np.array([1.0 for i in range(self.current_action.shape[0])]) - np.array(
            [-1.0 for i in range(self.current_action.shape[0])])) / (maximum - minimum)
        new_action = orig_minimum + scale * (self.current_action - minimum)
        action_penalty = np.sum(new_action ** 2) / new_action.shape[0]
        if self.task == 'reach':
            distance = physics.end_to_target()
            return rewards.tolerance(distance, bounds=(0, 0.01), margin=0.035) - 0.01 * action_penalty
        elif self.task == 'push':
            lfinger_to_target = physics.lfinger_to_target()
            lfinger_to_target_reward = rewards.tolerance(lfinger_to_target, bounds=(0, 0.001), margin=0.2)
            rfinger_to_target = physics.rfinger_to_target()
            rfinger_to_target_reward = rewards.tolerance(rfinger_to_target, bounds=(0, 0.001), margin=0.2)
            object_to_otarget = physics.object_to_otarget()
            object_to_otarget_reward = rewards.tolerance(object_to_otarget, bounds=(0, 0.01), margin=0.2)
            return (lfinger_to_target_reward + rfinger_to_target_reward + 2 * object_to_otarget_reward) / 4 - 0.01 * action_penalty
