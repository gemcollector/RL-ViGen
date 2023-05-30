# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from transfer_util import initialize_model
from stage1_models import BasicBlock, ResNet84
import os
import copy
from PIL import Image
import platform
from numbers import Number
import utils
from vrl3_agent import VRL3Agent, Actor, Critic, RandomShiftsAug


class CURLHead(nn.Module):
    def __init__(self, repr_dim):
        super().__init__()
        self.W = nn.Parameter(torch.rand(repr_dim, repr_dim))

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits



class CURLAgent(VRL3Agent):
    
    def __init__(self, aux_lr=0.3, aux_beta=0.9, **kwargs):
        super(CURLAgent, self).__init__(**kwargs)

        self.act_dim = kwargs['action_shape'][0]

        if kwargs['use_sensor']:
            downstream_input_dim = self.encoder.repr_dim + 24
        else:
            downstream_input_dim = self.encoder.repr_dim

        self.curl_head = CURLHead(downstream_input_dim).to(self.device)
        self.actor = Actor(downstream_input_dim, kwargs['action_shape'], kwargs['feature_dim'],
                           kwargs['hidden_dim']).to(self.device)

        self.critic = Critic(downstream_input_dim, kwargs['action_shape'], kwargs['feature_dim'],
                           kwargs['hidden_dim']).to(self.device)
        self.critic_target = Critic(downstream_input_dim, kwargs['action_shape'], kwargs['feature_dim'],
                           kwargs['hidden_dim']).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=kwargs['lr'])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=kwargs['lr'])
        self.curl_optimizer = torch.optim.Adam(
            self.curl_head.parameters(), lr=aux_lr, betas=(aux_beta, 0.999)
        )

        encoder_lr = kwargs['lr'] * kwargs['encoder_lr_scale']
        """ set up encoder optimizer """
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        # data augmentation
        self.aug = RandomShiftsAug(pad=4)
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        super().train(training)
        if hasattr(self, 'curl_head'):
            self.curl_head.train(training)

    def update_curl(self, z_a, z_pos):
        metrics = dict()
        # z_a = self.curl_head.encoder(x)
        # with torch.no_grad():
        #     z_pos = self.critic_target.encoder(x_pos)

        logits = self.curl_head.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        curl_loss = F.cross_entropy(logits, labels)

        self.curl_optimizer.zero_grad(set_to_none=True)
        self.encoder_opt.zero_grad(set_to_none=True)
        curl_loss.backward()
        self.curl_optimizer.step()
        self.encoder_opt.step()

        if self.use_tb:
            metrics['curl_loss'] = curl_loss.item()

        return metrics


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


        original_obs = obs.clone()
        pos_obs = obs.clone()
        original_obs = self.aug(original_obs.float())
        pos_obs = self.aug(pos_obs.float())

        # augment
        if self.use_data_aug:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
        else:
            obs = obs.float()
            next_obs = next_obs.float()

        # encode
        if update_encoder:
            obs = self.encoder(obs)
        else:
            with torch.no_grad():
                obs = self.encoder(obs)

        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        # concatenate obs with additional sensor observation if needed
        obs_combined = torch.cat([obs, obs_sensor], dim=1) if obs_sensor is not None else obs
        obs_next_combined = torch.cat([next_obs, obs_sensor_next], dim=1) if obs_sensor_next is not None else next_obs

        # update critic
        metrics.update(self.update_critic_vrl3(obs_combined, action, reward, discount, obs_next_combined,
                                               stddev, update_encoder, conservative_loss_weight))

        # update actor, following previous works, we do not use actor gradient for encoder update
        metrics.update(self.update_actor_vrl3(obs_combined.detach(), action, stddev, bc_weight,
                                              self.pretanh_penalty, self.pretanh_threshold))

        metrics['batch_reward'] = reward.mean().item()


        # update curl
        obs = self.encoder(original_obs)
        with torch.no_grad():
            pos = self.encoder(pos_obs)

        original_obs_combined = torch.cat([obs, obs_sensor], dim=1) if obs_sensor is not None else obs
        pos_obs_combined = torch.cat([pos, obs_sensor], dim=1) if obs_sensor is not None else pos


        metrics.update(self.update_curl(original_obs_combined, pos_obs_combined))

        # update critic target networks
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        return metrics
        