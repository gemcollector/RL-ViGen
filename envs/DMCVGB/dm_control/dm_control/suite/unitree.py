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
_DEFAULT_TIME_LIMIT = 10
_STAND_HEIGHT = 0.36
_WALK_SPEED = 1


def get_model_and_assets_walk():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model('mujoco_menagerie/unitree_a1/scene_walk.xml'), common.ASSETS


def get_model_and_assets_stand():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model('mujoco_menagerie/unitree_a1/scene_stand.xml'), common.ASSETS


@SUITE.add('benchmarking')
def walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns a Unitree that strives to stand upright, balancing its pose."""
    origin_dir = os.getcwd()
    current_dir = os.path.dirname(__file__)
    os.chdir(current_dir + '/mujoco_menagerie/unitree_a1')
    physics = Physics.from_xml_string(*get_model_and_assets_walk())
    os.chdir(origin_dir)
    task = Unitree(move_speed=_WALK_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


@SUITE.add('benchmarking')
def stand(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Stand task."""
    origin_dir = os.getcwd()
    current_dir = os.path.dirname(__file__)
    os.chdir(current_dir + '/mujoco_menagerie/unitree_a1')
    physics = Physics.from_xml_string(*get_model_and_assets_stand())
    os.chdir(origin_dir)
    task = Unitree(move_speed=0)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


class Physics(mujoco.Physics):

    def get_height(self):
        data = self.named.data
        height = data.xpos['trunk'][2]
        return height

    def get_upright(self):
        """Returns projection from z-axes of torso to the z-axes of world."""
        return self.named.data.xmat['trunk', 'zz']

    def get_forward(self):
        return self.named.data.xmat['trunk', 'xx']

    def horizontal_velocity(self):
        """Returns the horizontal velocity of the center-of-mass."""
        return self.named.data.sensordata['trunk_subtreelinvel'][0]

    def vertical_velocity(self):
        return self.named.data.sensordata['trunk_subtreelinvel'][2]

    def angular_velcity(self):
        return self.named.data.sensordata['trunk_frameangvel']

    def angular_xaxis(self):
        return self.named.data.sensordata['trunk_framexaxis']

    def angular_yaxis(self):
        return self.named.data.sensordata['trunk_frameyaxis']

    def angular_zaxis(self):
        return self.named.data.sensordata['trunk_framezaxis']


class Unitree(base.Task):

    def __init__(self, move_speed, random=None):
        """Initialize an instance of `Hopper`.

        Args:
          hopping: Boolean, if True the task is to hop forwards, otherwise it is to
            balance upright.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._move_speed = move_speed
        super(Unitree, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        physics.reset()
        # randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        # self._timeout_progress = 0
        if self._move_speed == 0:
            physics.data.qpos[-12:] = np.array(
                [0.05, 1.3, -2.69653, 0.05, 1.3, -2.69653, 0.1, 1.3, -2.69653, 0.1, 1.3, -2.69653])
        super(Unitree, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of positions, velocities and touch sensors."""
        obs = collections.OrderedDict()
        obs['position'] = physics.data.qpos[:].copy()
        obs['velocity'] = physics.data.qvel[:].copy()
        obs['height'] = physics.get_height()
        obs['horizontal_velocity'] = physics.horizontal_velocity()
        return obs

    def before_step(self, action, physics):
        """Sets the control signal for the actuators to values in `action`."""
        # Support legacy internal code.
        action = getattr(action, "continuous_actions", action)
        self.current_action = action
        physics.set_control(action)

    def get_reward(self, physics):
        angular_xaxis = physics.angular_xaxis()
        angular_yaxis = physics.angular_yaxis()
        angular_zaxis = physics.angular_zaxis()
        angular_xaxis_penalty = np.sum(np.square(angular_xaxis - np.array([0, 0, 1])))
        angular_yaxis_penalty = np.sum(np.square(angular_yaxis - np.array([0, -1, 0])))
        angular_zaxis_penalty = np.sum(np.square(angular_zaxis - np.array([1, 0, 0])))
        height = physics.get_height()
        standing = rewards.tolerance(height,
                                     bounds=(_STAND_HEIGHT, float('inf')),
                                     margin=_STAND_HEIGHT / 2)
        # stable_position = np.array([0,0.5,0, 0,0.5,0, 0,0.5,0, 0,0.5,0])
        # stable_position = np.array([0.05,0.22,-0.8, 0.05,0.22,-0.8, 0.1,0.68,0, 0.1,0.68,0])
        # action_penalty = np.sum((self.current_action - stable_position) ** 2) / 12

        # upright = (1 + physics.get_upright()) / 2
        upright = physics.get_upright()

        # Penalize z axis base linear velocity
        z_lin_vel_penalty = physics.vertical_velocity() ** 2
        # Penalize xy axes base angular velocity
        xy_ang_vel_penalty = np.sum(np.square(physics.angular_velcity()[:2]))

        """Returns a reward to the agent."""
        # if self._move_speed == 0:
        #   # - 0.05 * z_lin_vel_penalty
        #   stand_reward = (2 * standing + 1 * upright) / 3  - 0.05 * action_penalty - 0.1 * angular_xaxis_penalty - 0.05 * angular_zaxis_penalty - 0.005 * xy_ang_vel_penalty
        #   return stand_reward
        if self._move_speed == 0:
            minimum = self.action_spec(physics).minimum
            maximum = self.action_spec(physics).maximum
            orig_minimum = np.array([-1.0 for i in range(self.current_action.shape[0])])
            scale = (np.array([1.0 for i in range(self.current_action.shape[0])]) - np.array(
                [-1.0 for i in range(self.current_action.shape[0])])) / (maximum - minimum)
            new_action = orig_minimum + scale * (self.current_action - minimum)
            action_penalty = np.sum(new_action ** 2) / new_action.shape[0]
            # - 0.05 * z_lin_vel_penalty - 0.005 * xy_ang_vel_penalty
            stand_reward = (
                                       2 * standing + 1 * upright) / 3 - 0.08 * action_penalty - 0.1 * angular_xaxis_penalty - 0.05 * angular_zaxis_penalty
            return stand_reward
        else:
            stable_position = np.array([0.05, 0.22, -0.8, 0.05, 0.22, -0.8, 0.1, 0.68, 0, 0.1, 0.68, 0])
            action_penalty = np.sum((self.current_action - stable_position) ** 2) / 12
            forward = physics.get_forward()
            # - 0.05 * action_penalty
            stand_reward = (
                                       3 * standing + 1 * upright + 1 * forward) / 5 - 0.05 * action_penalty - 0.05 * z_lin_vel_penalty - 0.005 * xy_ang_vel_penalty - 0.1 * angular_xaxis_penalty - 0.05 * angular_zaxis_penalty
            move_reward = rewards.tolerance(physics.horizontal_velocity(), bounds=(self._move_speed, float('inf')),
                                            # margin=0.15,
                                            # margin=self._move_speed/2,
                                            # value_at_margin=0.5,
                                            margin=self._move_speed,
                                            sigmoid='linear'
                                            )
            # return stand_reward * (5*move_reward + 1) / 6
            return (stand_reward + move_reward + stand_reward * move_reward) / 3