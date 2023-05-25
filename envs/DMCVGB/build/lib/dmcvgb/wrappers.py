import numpy as np
from numpy.random import randint
import os
import gym
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import dmc2gym
from dmcvgb import utils
from collections import deque


class ColorWrapper(gym.Wrapper):
	"""Wrapper for the color experiments"""
	def __init__(self, env, background, seed=None, objects_color='original', table_texure='original', light_position='original', light_color='original', moving_light='original', cam_pos='original'):
		assert isinstance(env, FrameStack), 'wrapped env must be a framestack'
		gym.Wrapper.__init__(self, env)
		self._max_episode_steps = env._max_episode_steps
		self._background = background
		self._random_state = np.random.RandomState(seed)
		self.time_step = 0
		if self._background['type'] == 'color' or objects_color != 'original':
			self._load_colors()
		self._table_texure = table_texure
		self._objects_color = objects_color
		self._light_position = light_position
		self._light_color = light_color
		# self.model = self.env.env.env.env.env._env.physics.model
		self._moving_light = moving_light
		self._origin_light_pos = env.env.env.env.env._env.physics.model.light_pos
		self._origin_light_diffuse = env.env.env.env.env._env.physics.model.light_diffuse
		self._cam_pos = cam_pos
		# self._origin_cam_pos = env.env.env.env.env._env.physics.model.cam_pos
		# self._origin_cam_pos = env.env.env.env.env._env.physics.named.data.cam_xpos
		if self._moving_light == 'easy':
			self.step_pos = 0.01
			self.light_pos_range = [self._origin_light_pos[:, 1] - 5, self._origin_light_pos[:, 1] + 5]
		elif self._moving_light == 'hard':
			self.step_pos = 0.05
			self.light_pos_range = [self._origin_light_pos[:, 1] - 10, self._origin_light_pos[:, 1] + 10]

	def reset(self):
		self.time_step = 0
		setting_kwargs = {}
		if self._background['type'] == 'color':
			# self.randomize()
			# if self._background["difficulty"] != 'original':
			background_color = getattr(self, f'_colors_{self._background["difficulty"]}')[
				self._random_state.randint(len(getattr(self, f'_colors_{self._background["difficulty"]}')))]
			setting_kwargs['grid_rgb1'] = background_color['grid_rgb1']
			setting_kwargs['skybox_rgb'] = background_color['skybox_rgb']
			setting_kwargs['grid_rgb2'] = background_color['grid_rgb2']
		if self._objects_color != 'original':
			self_color = getattr(self, f'_colors_{self._objects_color}')[
				self._random_state.randint(len(getattr(self, f'_colors_{self._objects_color}')))]
			setting_kwargs['self_rgb'] = self_color['self_rgb']

		if self._background['type'] == 'video':
			# apply greenscreen
			# setting_kwargs = {
			# 	# 'skybox_rgb': [.2, .8, .2],
			# 	# 'skybox_rgb2': [.2, .8, .2],
			# 	# 'skybox_markrgb': [.2, .8, .2]
			# 	'skybox_rgb': [.0, .0, .0],
			# 	'skybox_rgb2': [.0, .0, .0],
			# 	'skybox_markrgb': [.0, .0, .0]
			# }
			setting_kwargs['skybox_rgb'] = [.0, .0, .0]
			setting_kwargs['skybox_rgb2'] = [.0, .0, .0]
			setting_kwargs['skybox_markrgb'] = [.0, .0, .0]

			if self._background['difficulty'] == 'hard':
				# setting_kwargs['grid_rgb1'] = [.2, .8, .2]
				# setting_kwargs['grid_rgb2'] = [.2, .8, .2]
				# setting_kwargs['grid_markrgb'] = [.2, .8, .2]
				setting_kwargs['grid_rgb1'] = [.0, .0, .0]
				setting_kwargs['grid_rgb2'] = [.0, .0, .0]
				setting_kwargs['grid_markrgb'] = [.0, .0, .0]
		self.reload_physics(setting_kwargs)
		if self._light_position == 'easy':
			self.env.env.env.env.env._env.physics.model.light_pos = self._origin_light_pos + self._random_state.randint(3,5,size=self._origin_light_pos.shape)
		elif self._light_position == 'hard':
			self.env.env.env.env.env._env.physics.model.light_pos = self._origin_light_pos + self._random_state.randint(9,11,size=self._origin_light_pos.shape)
		if self._light_color == 'easy':
			self.env.env.env.env.env._env.physics.model.light_diffuse = self._origin_light_diffuse + self._random_state.uniform(-0.2,0.2,size=self._origin_light_diffuse.shape)
		elif self._light_color == 'hard':
			# self.env.env.env.env.env._env.physics.model.light_diffuse = np.array([0.2, 0.7, 0.7])
			self.env.env.env.env.env._env.physics.model.light_diffuse = self._origin_light_diffuse + self._random_state.uniform(-10,10,size=self._origin_light_diffuse.shape)

		return self.env.reset()

	def step(self, action):
		self.time_step += 1
		if self._moving_light != 'original':
			self.env.env.env.env.env._env.physics.model.light_pos[:, :] += self.step_pos
			if self.env.env.env.env.env._env.physics.model.light_pos[:, 0].all() > self.light_pos_range[1].all() or self.env.env.env.env.env._env.physics.model.light_pos[:, 0].all() < \
					self.light_pos_range[0].all():
				self.step_pos *= -1
		return self.env.step(action)

	def randomize(self):
		# assert 'color' in self._mode, f'can only randomize in color mode, received {self._mode}'
		self.reload_physics(self.get_random_color())

	def _load_colors(self):
		# assert self._mode in {'color_easy', 'color_hard'}
		self._colors_easy = torch.load(f'{os.path.dirname(__file__)}/../data/color_easy.pt')
		self._colors_hard = torch.load(f'{os.path.dirname(__file__)}/../data/color_hard.pt')

	def get_random_color(self):
		# assert len(self._colors) >= 100, 'env must include at least 100 colors'
		color = {}
		if self._background['type'] != 'original':
			background_color = getattr(self, f'_colors_{self._background["difficulty"]}')[self._random_state.randint(len(getattr(self, f'_colors_{self._background["difficulty"]}')))]
			color['grid_rgb1'] = background_color['grid_rgb1']
			color['skybox_rgb'] = background_color['skybox_rgb']
			color['grid_rgb2'] = background_color['grid_rgb2']
		if self._objects_color != 'original':
			self_color = getattr(self, f'_colors_{self._objects_color}')[self._random_state.randint(len(getattr(self, f'_colors_{self._objects_color}')))]
			color['self_rgb'] = self_color['self_rgb']
		# yang = self_color
		return color

	def reload_physics(self, setting_kwargs=None, state=None):
		from dm_control.suite import common
		domain_name = self._get_dmc_wrapper()._domain_name
		if domain_name == 'unitree':
			domain_name = 'mujoco_menagerie/unitree_a1/scene_' + self._get_dmc_wrapper()._task_name
			if self._table_texure != 'original':
				setting_kwargs['ground_texture'] = 'table_' + self._table_texure + str(self._random_state.randint(10))
		if domain_name == 'franka':
			domain_name = 'mujoco_menagerie/franka_emika_panda/scene_' + self._get_dmc_wrapper()._task_name
			if self._table_texure != 'original':
				setting_kwargs['table_texture'] = 'table_' + self._table_texure + str(self._random_state.randint(10))
			if self._objects_color != 'original':
				setting_kwargs['self_rgb1'] = None
		elif domain_name == 'quadruped':
			domain_name = domain_name + '_' + self._get_dmc_wrapper()._task_name
		if setting_kwargs is None:
			setting_kwargs = {}
		# if state is None:
		# 	state = self._get_state()

		self._reload_physics(
			*common.settings.get_model_and_assets_from_setting_kwargs(
				domain_name+'.xml', setting_kwargs
			)
		)
		# self._set_state(state)
	
	def get_state(self):
		return self._get_state()
	
	def set_state(self, state):
		self._set_state(state)

	def _get_dmc_wrapper(self):
		_env = self.env
		while not isinstance(_env, dmc2gym.wrappers.DMCWrapper) and hasattr(_env, 'env'):
			_env = _env.env
		assert isinstance(_env, dmc2gym.wrappers.DMCWrapper), 'environment is not dmc2gym-wrapped'

		return _env

	def _reload_physics(self, xml_string, assets=None):
		_env = self.env
		while not hasattr(_env, '_physics') and hasattr(_env, 'env'):
			_env = _env.env
		assert hasattr(_env, '_physics'), 'environment does not have physics attribute'
		key_list = list(assets.keys())
		value_list = list(assets.values())
		for i in range(len(key_list)):
			if type(assets[key_list[i]]).__name__ == 'str':
				assets[key_list[i]] = bytes(value_list[i], 'utf-8')

		_env.physics.reload_from_xml_string(xml_string, assets=assets, domain_name=_env._domain_name)

	def _get_physics(self):
		_env = self.env
		while not hasattr(_env, '_physics') and hasattr(_env, 'env'):
			_env = _env.env
		assert hasattr(_env, '_physics'), 'environment does not have physics attribute'

		return _env._physics

	def _get_state(self):
		return self._get_physics().get_state()
		
	def _set_state(self, state):
		self._get_physics().set_state(state)


class FrameStack(gym.Wrapper):
	"""Stack frames as observation"""
	def __init__(self, env, k):
		gym.Wrapper.__init__(self, env)
		self._k = k
		self._frames = deque([], maxlen=k)
		shp = env.observation_space.shape
		self.observation_space = gym.spaces.Box(
			low=0,
			high=1,
			shape=((shp[0] * k,) + shp[1:]),
			dtype=env.observation_space.dtype
		)
		self._max_episode_steps = env._max_episode_steps

	def reset(self):
		obs = self.env.reset()
		for _ in range(self._k):
			self._frames.append(obs)
		return self._get_obs()

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		# obs, reward, done, _, info = self.env.step(action)  # yangsizhe
		self._frames.append(obs)
		return self._get_obs(), reward, done, info

	def _get_obs(self):
		assert len(self._frames) == self._k
		return utils.LazyFrames(list(self._frames))


def rgb_to_hsv(r, g, b):
	"""Convert RGB color to HSV color"""
	maxc = max(r, g, b)
	minc = min(r, g, b)
	v = maxc
	if minc == maxc:
		return 0.0, 0.0, v
	s = (maxc-minc) / maxc
	rc = (maxc-r) / (maxc-minc)
	gc = (maxc-g) / (maxc-minc)
	bc = (maxc-b) / (maxc-minc)
	if r == maxc:
		h = bc-gc
	elif g == maxc:
		h = 2.0+rc-bc
	else:
		h = 4.0+gc-rc
	h = (h/6.0) % 1.0
	return h, s, v

import matplotlib.pyplot as plt
def do_green_screen(x, bg):
	"""Removes green background from observation and replaces with bg; not optimized for speed"""
	assert isinstance(x, np.ndarray) and isinstance(bg, np.ndarray), 'inputs must be numpy arrays'
	assert x.dtype == np.uint8 and bg.dtype == np.uint8, 'inputs must be uint8 arrays'
	# plt.figure(dpi=300)
	# plt.imshow(x.swapaxes(0, 1).swapaxes(1, 2))
	# plt.show()
	# Get image sizes
	x_h, x_w = x.shape[1:]

	# Convert to RGBA images
	im = TF.to_pil_image(torch.ByteTensor(x))
	im = im.convert('RGBA')
	pix = im.load()
	bg = TF.to_pil_image(torch.ByteTensor(bg))
	bg = bg.convert('RGBA')
	bg = bg.load()

	# Replace pixels
	for x in range(x_w):
		for y in range(x_h):
			r, g, b, a = pix[x, y]
			h_ratio, s_ratio, v_ratio = rgb_to_hsv(r / 255., g / 255., b / 255.)
			h, s, v = (h_ratio * 360, s_ratio * 255, v_ratio * 255)

			# min_h, min_s, min_v = (100, 80, 70)
			# max_h, max_s, max_v = (185, 255, 255)
			# min_h, min_s, min_v = (130, 110, 100)
			# max_h, max_s, max_v = (155, 225, 225)
			# if min_h <= h <= max_h and min_s <= s <= max_s and min_v <= v <= max_v:  # 替换一定范围内的rgb
			if r == g == b == 0:
				pix[x, y] = bg[x, y]

	return np.moveaxis(np.array(im).astype(np.uint8), -1, 0)[:3]


class VideoWrapper(gym.Wrapper):
	"""Green screen for video experiments"""
	def __init__(self, env, background, seed, objects_color='original', cam_pos='original'):
		gym.Wrapper.__init__(self, env)
		self._background = background
		self._seed = seed
		self._random_state = np.random.RandomState(seed)
		self._index = 0
		self._video_paths = []
		if self._background['type'] == 'video':
			self._get_video_paths()
		self._num_videos = len(self._video_paths)
		self._max_episode_steps = env._max_episode_steps
		# self._cam_pos = cam_pos
		# self._origin_cam_pos = env.env.env.env.env._env.physics.named.data.cam_xpos

	def _get_video_paths(self):
		current_dir = os.path.dirname(__file__)
		video_dir = os.path.join(f'{current_dir}/../data', f'video_{self._background["difficulty"]}')
		if self._background['difficulty'] == 'easy':
			self._video_paths = [os.path.join(video_dir, f'video{i}.mp4') for i in range(10)]
		elif self._background['difficulty'] == 'hard':
			self._video_paths = [os.path.join(video_dir, f'video{i}.mp4') for i in range(100)]
		# else:
		# 	raise ValueError(f'received unknown mode "{self._mode}"')

	def _load_video(self, video):
		"""Load video from provided filepath and return as numpy array"""
		import cv2
		cap = cv2.VideoCapture(video)
		assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) >= 100, 'width must be at least 100 pixels'
		assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) >= 100, 'height must be at least 100 pixels'
		n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		buf = np.empty((n, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), np.dtype('uint8'))
		i, ret = 0, True
		while (i < n  and ret):
			ret, frame = cap.read()
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			buf[i] = frame
			i += 1
		cap.release()
		# yang = np.moveaxis(buf, -1, 1)
		return np.moveaxis(buf, -1, 1)

	def _reset_video(self):
		self._index = (self._index + 1) % self._num_videos
		self._data = self._load_video(self._video_paths[self._index])

	def reset(self):
		if self._background['type'] == 'video':
			self._reset_video()
		self._current_frame = 0

		# return self.env.reset()
		return self._greenscreen(self.env.reset())

	def step(self, action):
		self._current_frame += 1
		obs, reward, done, info = self.env.step(action)
		# obs, reward, done, _, info = self.env.step(action)  # yangsizhe
		return self._greenscreen(obs), reward, done, info
	
	def _interpolate_bg(self, bg, size:tuple):
		"""Interpolate background to size of observation"""
		bg = torch.from_numpy(bg).float().unsqueeze(0)/255.
		bg = F.interpolate(bg, size=size, mode='bilinear', align_corners=False)
		return (bg*255.).byte().squeeze(0).numpy()

	def _greenscreen(self, obs):
		"""Applies greenscreen if video is selected, otherwise does nothing"""
		if self._background['type'] == 'video':
			bg = self._data[self._current_frame % len(self._data)] # select frame
			bg = self._interpolate_bg(bg, obs.shape[1:]) # scale bg to observation size
			return do_green_screen(obs, bg) # apply greenscreen
		return obs

	def apply_to(self, obs):
		"""Applies greenscreen mode of object to observation"""
		obs = obs.copy()
		channels_last = obs.shape[-1] == 3
		if channels_last:
			obs = torch.from_numpy(obs).permute(2,0,1).numpy()
		obs = self._greenscreen(obs)
		if channels_last:
			obs = torch.from_numpy(obs).permute(1,2,0).numpy()
		return obs
