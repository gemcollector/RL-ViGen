# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)



from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


import json
import os
import torchvision.transforms as TF
import torchvision.datasets as datasets

places_dataloader = None
places_iter = None

def load_config(key=None):
    current_dir = os.path.dirname(__file__)
    path = os.path.join(f'{current_dir}/cfgs', 'aug_config.cfg')
    with open(path) as f:
        data = json.load(f)
    if key is not None:
        return data[key]
    return data


def _load_places(batch_size=256, image_size=84, num_workers=8, use_val=False):
	global places_dataloader, places_iter
	partition = 'val' if use_val else 'train'
	print(f'Loading {partition} partition of places365_standard...')
	for data_dir in load_config('datasets'):
		if os.path.exists(data_dir):
			fp = os.path.join(data_dir, 'places365_standard', partition)
			if not os.path.exists(fp):
				print(f'Warning: path {fp} does not exist, falling back to {data_dir}')
				fp = data_dir
			places_dataloader = torch.utils.data.DataLoader(
				datasets.ImageFolder(fp, TF.Compose([
					TF.RandomResizedCrop(image_size),
					TF.RandomHorizontalFlip(),
					TF.ToTensor()
				])),
				batch_size=batch_size, shuffle=True,
				num_workers=num_workers, pin_memory=True)
			places_iter = iter(places_dataloader)
			break
	if places_iter is None:
		raise FileNotFoundError('failed to find places365 data at any of the specified paths')
	print('Loaded dataset from', data_dir)


def _get_places_batch(batch_size):
	global places_iter
	try:
		imgs, _ = next(places_iter)
		if imgs.size(0) < batch_size:
			places_iter = iter(places_dataloader)
			imgs, _ = next(places_iter)
	except StopIteration:
		places_iter = iter(places_dataloader)
		imgs, _ = next(places_iter)
	return imgs.cuda()


def random_overlay(x, dataset='places365_standard'):
	"""Randomly overlay an image from Places"""
	global places_iter
	alpha = 0.5

	if dataset == 'places365_standard':
		if places_dataloader is None:
			_load_places(batch_size=x.size(0), image_size=x.size(-1))
		imgs = _get_places_batch(batch_size=x.size(0)).repeat(1, x.size(1)//3, 1, 1)
	else:
		raise NotImplementedError(f'overlay has not been implemented for dataset "{dataset}"')

	return ((1-alpha)*(x/255.) + (alpha)*imgs)*255.


def cat(x, y, axis=0):
	return torch.cat([x, y], axis=0)


def attribution_augmentation(x, mask, dataset="places365_standard"):
    """Complete non importnant pixels with a random image from Places"""
    global places_iter

    if dataset == "places365_standard":
        if places_dataloader is None:
            _load_places(batch_size=x.size(0), image_size=x.size(-1))
        imgs = _get_places_batch(batch_size=x.size(0)).repeat(1, x.size(1) // 3, 1, 1)
    else:
        raise NotImplementedError(
            f'overlay has not been implemented for dataset "{dataset}"'
        )

    # s_plus = random_conv(x) * mask
    s_plus = x * mask
    s_tilde = (((s_plus) / 255.0) + (imgs * (torch.ones_like(mask) - mask))) * 255.0
    s_minus = imgs * 255
    return s_tilde



# The SRM code, version 1, circle-ring shaped mask
def random_mask_freq_v1(x):
        p = random.uniform(0, 1)
        if p > 0.5:
             return x
        # need to adjust r1 r2 and delta for best performance
        r1=random.uniform(0,0.5)
        delta_r=random.uniform(0,0.035)
        r2=np.min((r1+delta_r,0.5))
        # print(r2)
        # generate Mask M
        B,C,H,W = x.shape
        center = (int(H/2), int(W/2))
        diagonal_lenth = max(H,W) # np.sqrt(H**2+W**2) is also ok, use a smaller r1
        r1_pix = diagonal_lenth * r1
        r2_pix = diagonal_lenth * r2
        Y_coord, X_coord = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((Y_coord - center[0])**2 + (X_coord - center[1])**2)
        M = dist_from_center <= r2_pix
        M = M * (dist_from_center >= r1_pix)
        M = ~M

        # mask Fourier spectrum
        M = torch.from_numpy(M).float().to(x.device)
        srm_out = torch.zeros_like(x)
        for i in range(C):
            x_c = x[:,i,:,:]
            x_spectrum = torch.fft.fftn(x_c, dim=(-2,-1))
            x_spectrum = torch.fft.fftshift(x_spectrum, dim=(-2,-1))
            out_spectrum = x_spectrum * M
            out_spectrum = torch.fft.ifftshift(out_spectrum, dim=(-2,-1))
            srm_out[:,i,:,:] = torch.fft.ifftn(out_spectrum, dim=(-2,-1)).float()
        return srm_out


def random_mask_freq_v2(x):
    p = random.uniform(0, 1)
    if p > 0.5:
        return x

    # dynamicly select freq range to erase
    A = 0
    B = 0.5
    a = random.uniform(A, B)
    C = 2
    freq_limit_low = round(a, C)

    A = 0
    B = 0.05
    a = random.uniform(A, B)
    C = 2
    diff = round(a, C)
    freq_limit_hi = freq_limit_low + diff

    # b, 9, h, w
    b, c, h, w = x.shape
    x0, x1, x2 = torch.chunk(x, 3, dim=1)
    # b, 3, 3, h, w
    x = torch.cat((x0.unsqueeze(1), x1.unsqueeze(1), x2.unsqueeze(1)), dim=1)

    pass1 = torch.abs(torch.fft.fftfreq(x.shape[-1], device=x.device)) < freq_limit_hi
    pass2 = torch.abs(torch.fft.fftfreq(x.shape[-2], device=x.device)) < freq_limit_hi
    kernel1 = torch.outer(pass2, pass1)  # freq_limit_hi square is true

    pass1 = torch.abs(torch.fft.fftfreq(x.shape[-1], device=x.device)) < freq_limit_low
    pass2 = torch.abs(torch.fft.fftfreq(x.shape[-2], device=x.device)) < freq_limit_low
    kernel2 = torch.outer(pass2, pass1)  # freq_limit_low square is true

    kernel = kernel1 * (~kernel2)  # a square ring is true
    fft_1 = torch.fft.fftn(x, dim=(2, 3, 4))
    imgs = torch.fft.ifftn(fft_1 * (~kernel), dim=(2, 3, 4)).float()
    x0, x1, x2 = torch.chunk(imgs, 3, dim=1)
    imgs = torch.cat((x0.squeeze(1), x1.squeeze(1), x2.squeeze(1)), dim=1)

    return imgs


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path