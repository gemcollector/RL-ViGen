# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34
from torchvision import transforms
import utils
from utils import random_overlay
from vrl3_agent import VRL3Agent, Actor, Critic, RandomShiftsAug



class ResEncoder(nn.Module):
    def __init__(self):
        super(ResEncoder, self).__init__()
        self.model = resnet18(pretrained=True)
        self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224)
            ])

        for param in self.model.parameters():
            param.requires_grad = False

        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.repr_dim = 1024
        self.image_channel = 3
        x = torch.randn([32] + [9, 84, 84])
        with torch.no_grad():
            out_shape = self.forward_conv(x).shape
        self.out_dim = out_shape[1]
        self.fc = nn.Linear(self.out_dim, self.repr_dim)
        self.ln = nn.LayerNorm(self.repr_dim)
        self.n_images = 3
        #
        # # Initialization
        # nn.init.orthogonal_(self.fc.weight.data)
        # self.fc.bias.data.fill_(0.0)

    @torch.no_grad()
    def forward_conv(self, obs, flatten=True):
        obs = obs / 255.0 - 0.5
        time_step = obs.shape[1] // self.image_channel
        obs = obs.view(obs.shape[0], time_step, self.image_channel, obs.shape[-2], obs.shape[-1])
        obs = obs.view(obs.shape[0] * time_step, self.image_channel, obs.shape[-2], obs.shape[-1])

        for name, module in self.model._modules.items():
            obs = module(obs)
            if name == 'layer2':
                break

        conv = obs.view(obs.size(0) // time_step, time_step, obs.size(1), obs.size(2), obs.size(3))
        conv_current = conv[:, 1:, :, :, :]
        conv_prev = conv_current - conv[:, :time_step - 1, :, :, :].detach()
        conv = torch.cat([conv_current, conv_prev], axis=1)
        conv = conv.view(conv.size(0), conv.size(1) * conv.size(2), conv.size(3), conv.size(4))
        if flatten:
            conv = conv.view(conv.size(0), -1)

        return conv


    def forward(self, obs):
        conv = self.forward_conv(obs)
        out = self.fc(conv)
        out = self.ln(out)
        # obs = self.model(self.transform(obs.to(torch.float32)) / 255.0 - 0.5)
        return out

    def expand_first_layer(self):
        # convolutional channel expansion to deal with input mismatch
        pass


class PIEGAgent(VRL3Agent):
    def __init__(self, **kwargs):
        super(PIEGAgent, self).__init__(**kwargs)
        self.encoder = ResEncoder().to(kwargs['device'])
        self.device = kwargs['device']


        if kwargs['use_sensor']:
            downstream_input_dim = self.encoder.repr_dim + 24
        else:
            downstream_input_dim = self.encoder.repr_dim

        self.actor = Actor(downstream_input_dim, kwargs['action_shape'], kwargs['feature_dim'], kwargs['hidden_dim']).to(kwargs['device'])
        self.critic = Critic(downstream_input_dim, kwargs['action_shape'], kwargs['feature_dim'], kwargs['hidden_dim']).to(kwargs['device'])
        self.critic_target = Critic(downstream_input_dim, kwargs['action_shape'],
                                    kwargs['feature_dim'], kwargs['hidden_dim']).to(kwargs['device'])
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=kwargs['lr'])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=kwargs['lr'])

        encoder_lr = kwargs['lr'] * kwargs['encoder_lr_scale']
        """ set up encoder optimizer """
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        # data augmentation
        self.aug = RandomShiftsAug(pad=4)
        self.train()
        self.critic_target.train()


    def update_critic_vrl3(self, obs, action, reward, discount, next_obs, stddev, update_encoder, conservative_loss_weight, aug_obs):
        metrics = dict()
        batch_size = obs.shape[0]

        """
        STANDARD Q LOSS COMPUTATION:
        - get standard Q loss first, this is the same as in any other online RL methods
        - except for the safe Q technique, which controls how large the Q value can be
        """
        with torch.no_grad():
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

            if self.safe_q_target_factor < 1:
                target_Q[target_Q > (self.q_threshold + 1)] = self.q_threshold + (target_Q[target_Q > (self.q_threshold+1)] - self.q_threshold) ** self.safe_q_target_factor

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        aug_Q1, aug_Q2 = self.critic(aug_obs, action)
        aug_loss = F.mse_loss(aug_Q1, target_Q) + F.mse_loss(aug_Q2, target_Q)

        critic_loss = 0.5 * (critic_loss + aug_loss)

        """
        CONSERVATIVE Q LOSS COMPUTATION:
        - sample random actions, actions from policy and next actions from policy, as done in CQL authors' code
          (though this detail is not really discussed in the CQL paper)
        - only compute this loss when conservative loss weight > 0
        """
        if conservative_loss_weight > 0:
            random_actions = (torch.rand((batch_size * self.cql_n_random, self.act_dim), device=self.device) - 0.5) * 2

            dist = self.actor(obs, stddev)
            current_actions = dist.sample(clip=self.stddev_clip)

            dist = self.actor(next_obs, stddev)
            next_current_actions = dist.sample(clip=self.stddev_clip)

            # now get Q values for all these actions (for both Q networks)
            obs_repeat = obs.unsqueeze(1).repeat(1, self.cql_n_random, 1).view(obs.shape[0] * self.cql_n_random,
                                                                               obs.shape[1])

            Q1_rand, Q2_rand = self.critic(obs_repeat,
                                           random_actions)  # TODO might want to double check the logic here see if the repeat is correct
            Q1_rand = Q1_rand.view(obs.shape[0], self.cql_n_random)
            Q2_rand = Q2_rand.view(obs.shape[0], self.cql_n_random)

            Q1_curr, Q2_curr = self.critic(obs, current_actions)
            Q1_curr_next, Q2_curr_next = self.critic(obs, next_current_actions)

            # now concat all these Q values together
            Q1_cat = torch.cat([Q1_rand, Q1, Q1_curr, Q1_curr_next], 1)
            Q2_cat = torch.cat([Q2_rand, Q2, Q2_curr, Q2_curr_next], 1)

            cql_min_q1_loss = torch.logsumexp(Q1_cat / self.cql_temp,
                                              dim=1, ).mean() * conservative_loss_weight * self.cql_temp
            cql_min_q2_loss = torch.logsumexp(Q2_cat / self.cql_temp,
                                              dim=1, ).mean() * conservative_loss_weight * self.cql_temp

            """Subtract the log likelihood of data"""
            conservative_q_loss = cql_min_q1_loss + cql_min_q2_loss - (Q1.mean() + Q2.mean()) * conservative_loss_weight
            critic_loss_combined = critic_loss + conservative_q_loss
        else:
            critic_loss_combined = critic_loss

        # logging
        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        # if needed, also update encoder with critic loss
        if update_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss_combined.backward()
        self.critic_opt.step()
        if update_encoder:
            self.encoder_opt.step()

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
            aug_obs = self.encoder(random_overlay(original_obs, self.device))
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
