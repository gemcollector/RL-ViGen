# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from utils import random_overlay

from svea import SVEAAgent
from utils import random_mask_freq_v2, random_mask_freq_v1



class SRMAgent(SVEAAgent):

    def update(self, replay_iter, step, stage, use_sensor):
        # for stage 2 and 3, we use the same functions but with different hyperparameters
        assert stage in (2, 3)
        metrics = dict()

        if stage == 2:
            update_encoder = self.stage2_update_encoder
            stddev = self.stage2_std
            conservative_loss_weight = self.cql_weight
            bc_weight = self.stage2_bc_weight

        if stage == 3:
            if step % self.update_every_steps != 0:
                return metrics
            update_encoder = self.stage3_update_encoder

            stddev = utils.schedule(self.stddev_schedule, step)
            conservative_loss_weight = 0

            # compute stage 3 BC weight
            bc_data_per_iter = 40000
            i_iter = step // bc_data_per_iter
            bc_weight = self.stage3_bc_lam0 * self.stage3_bc_lam1 ** i_iter

        # batch data
        batch = next(replay_iter)
        if use_sensor: # TODO might want to...?
            obs, action, reward, discount, next_obs, obs_sensor, obs_sensor_next = utils.to_torch(batch, self.device)
        else:
            obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)
            obs_sensor, obs_sensor_next = None, None

        # augment
        if self.use_data_aug:
            obs = self.aug(obs.float())
            original_obs = obs.clone()
            next_obs = self.aug(next_obs.float())
        else:
            obs = obs.float()
            next_obs = next_obs.float()

        # encode
        if update_encoder:
            obs = self.encoder(obs)
            # strong augmentation
            aug_obs = self.encoder(random_mask_freq_v2(random_overlay(original_obs, self.device)))
        else:
            with torch.no_grad():
                obs = self.encoder(obs)

        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        # concatenate obs with additional sensor observation if needed
        obs_combined = torch.cat([obs, obs_sensor], dim=1) if obs_sensor is not None else obs
        obs_next_combined = torch.cat([next_obs, obs_sensor_next], dim=1) if obs_sensor_next is not None else next_obs
        aug_obs_combined = torch.cat([aug_obs, obs_sensor], dim=1) if obs_sensor is not None else aug_obs

        # update critic
        metrics.update(self.update_critic_vrl3(obs_combined, action, reward, discount, obs_next_combined,
                                               stddev, update_encoder, conservative_loss_weight, aug_obs_combined))

        # update actor, following previous works, we do not use actor gradient for encoder update
        metrics.update(self.update_actor_vrl3(obs_combined.detach(), action, stddev, bc_weight,
                                              self.pretanh_penalty, self.pretanh_threshold))

        metrics['batch_reward'] = reward.mean().item()

        # update critic target networks
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        return metrics

