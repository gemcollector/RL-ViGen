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
from rl_utils import (
    compute_attribution,
    compute_attribution_mask,
    make_attribution_pred_grid,
    make_obs_grid,
    make_obs_grad_grid,
)
from vrl3_agent import VRL3Agent, Actor, Critic
from utils import random_overlay

import random



class AttributionDecoder(nn.Module):
    def __init__(self,action_shape, emb_dim=100) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features=emb_dim+action_shape, out_features=32*21*21)
        self.conv1 = nn.Conv2d(
            in_channels=32, out_channels=128, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=9, kernel_size=3, padding=1)

    def forward(self, x, action):
        x = torch.cat([x,action],dim=1)
        x = self.proj(x).view(-1, 32, 21, 21)
        x = self.relu(x)
        x = self.conv1(x)
        x = F.upsample(x, scale_factor=2)
        x = self.relu(x)
        x = self.conv2(x)
        x = F.upsample(x, scale_factor=2)
        x = self.relu(x)
        x = self.conv3(x)
        return x

class AttributionPredictor(nn.Module):
    def __init__(self, action_shape, encoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = AttributionDecoder(action_shape, encoder.repr_dim)

    def forward(self, x, action):
        x = self.encoder(x)
        return self.decoder(x, action)


class SGQNAgent(VRL3Agent):

    def __init__(self, aux_lr=0.3, aux_beta=0.9, sgqn_quantile=0.95, **kwargs):
        super().__init__(**kwargs)

        self.attribution_predictor = AttributionPredictor(kwargs['action_shape'][0],
                                                          self.encoder).to(self.device)

        self.quantile = sgqn_quantile

        if kwargs['use_sensor']:
            downstream_input_dim = self.encoder.repr_dim + 24
        else:
            downstream_input_dim = self.encoder.repr_dim

        self.actor = Actor(downstream_input_dim, kwargs['action_shape'], kwargs['feature_dim'],
                           kwargs['hidden_dim']).to(self.device)

        self.critic = Critic(downstream_input_dim, kwargs['action_shape'], kwargs['feature_dim'],
                           kwargs['hidden_dim']).to(self.device)
        self.critic_target = Critic(downstream_input_dim, kwargs['action_shape'], kwargs['feature_dim'],
                           kwargs['hidden_dim']).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=kwargs['lr'])
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=kwargs['lr'])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=kwargs['lr'])
        self.aux_optimizer = torch.optim.Adam(
            self.attribution_predictor.parameters(),
            lr=aux_lr,
            betas=(aux_beta, 0.999),
        )

        self.train()
        self.critic_target.train()

    def compute_attribution_loss(self, obs, action, mask):
        mask = mask.float()
        attrib = self.attribution_predictor(obs.detach(), action.detach())
        aux_loss = F.binary_cross_entropy_with_logits(attrib, mask.detach())
        return attrib, aux_loss

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

        masked_Q1, masked_Q2 = self.critic(aug_obs, action)
        critic_loss += 0.5 * (F.mse_loss(Q1, masked_Q1) + F.mse_loss(Q2, masked_Q2))

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


    def update_aux(self, obs, obs_sensor, action):
        # obs: augmented obs
        # mask = compute_attribution_mask(obs_grad, self.quantile)
        obs_grad = compute_attribution(self.encoder, self.critic, obs, obs_sensor, action.detach())
        mask = compute_attribution_mask(obs_grad, self.quantile)
        s_tilde = random_overlay(obs.clone(), self.device)
        self.aux_optimizer.zero_grad()
        pred_attrib, aux_loss = self.compute_attribution_loss(s_tilde, action, mask)
        aux_loss.backward()
        self.aux_optimizer.step()


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
            aug_obs = obs.clone()
            next_obs = self.aug(next_obs.float())
        else:
            obs = obs.float()
            next_obs = next_obs.float()


        obs_grad = compute_attribution(self.encoder, self.critic, obs, obs_sensor, action.detach())
        mask = compute_attribution_mask(obs_grad, self.quantile)
        masked_obs = obs * mask
        masked_obs[mask < 1] = random.uniform(obs.view(-1).min(), obs.view(-1).max())

        masked_obs = self.encoder(masked_obs)

        # strong augmentation
        # aug_obs = self.encoder(random_overlay(original_obs, self.device))

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
        masked_obs_combined = torch.cat([masked_obs, obs_sensor], dim=1) if obs_sensor is not None else masked_obs




        # update critic
        metrics.update(self.update_critic_vrl3(obs_combined, action, reward, discount, obs_next_combined,
                                               stddev, update_encoder, conservative_loss_weight, masked_obs_combined))

        # update actor, following previous works, we do not use actor gradient for encoder update
        metrics.update(self.update_actor_vrl3(obs_combined.detach(), action, stddev, bc_weight,
                                              self.pretanh_penalty, self.pretanh_threshold))

        metrics['batch_reward'] = reward.mean().item()

        # update critic target networks
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        self.update_aux(aug_obs, obs_sensor, action)


        return metrics





