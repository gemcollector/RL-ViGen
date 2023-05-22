import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from rl_utils import (
    compute_attribution,
    compute_attribution_mask,
    make_attribution_pred_grid,
    make_obs_grid,
    make_obs_grad_grid,
)
from drqv2 import DrQV2Agent, Actor, Critic
from utils import attribution_augmentation, random_overlay
import random

def _get_out_shape(in_shape, layers):
    x = torch.randn(*in_shape).unsqueeze(0)
    return layers(x).squeeze(0).shape

class SharedCNN(nn.Module):
    def __init__(self, obs_shape, num_layers=11, num_filters=32):
        super().__init__()
        assert len(obs_shape) == 3
        self.num_layers = num_layers
        self.num_filters = num_filters

        self.layers = [
            nn.Conv2d(obs_shape[0], num_filters, 3, stride=2),
        ]
        for _ in range(1, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers = nn.Sequential(*self.layers)
        self.out_shape = _get_out_shape(obs_shape, self.layers)
        self.apply(utils.weight_init)

    def forward(self, x):
        x = x / 255.0 - 0.5
        return self.layers(x)

class SACEncoder(nn.Module):
    def __init__(self, shared_cnn, projection=None):
        super().__init__()
        self.shared_cnn = shared_cnn
        self.embed_dim = 32 * 21 * 21
        self.repr_dim = 512
        self.projection = nn.Linear(self.embed_dim, self.repr_dim)

    def forward(self, x):
        x = self.shared_cnn(x)
        x = x.view(x.shape[0], -1)
        return self.projection(x)


class CNNEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.embed_dim = 32 * 35 * 35
        self.repr_dim = 512

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        self.projector = nn.Linear(self.embed_dim, self.repr_dim)
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        h = self.projector(h)
        return h



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

    def forward(self, x,action):
        x = self.encoder(x)
        return self.decoder(x, action)


class SGQNAgent(DrQV2Agent):
    def __init__(self, aux_lr=0.3, aux_beta=0.9, sgqn_quantile=0.95, **kwargs):
        super().__init__(**kwargs)
        # shared_cnn = SharedCNN(kwargs['obs_shape']).to(self.device)
        # self.encoder = SACEncoder(shared_cnn).to(self.device)
        self.encoder = CNNEncoder(kwargs['obs_shape']).to(self.device)
        self.attribution_predictor = AttributionPredictor(kwargs['action_shape'][0],
                                                          self.encoder).to(self.device)
        self.quantile = sgqn_quantile

        self.actor = Actor(self.encoder.repr_dim, kwargs['action_shape'], kwargs['feature_dim'],
                           kwargs['hidden_dim']).to(self.device)

        self.critic = Critic(self.encoder.repr_dim, kwargs['action_shape'], kwargs['feature_dim'],
                           kwargs['hidden_dim']).to(self.device)
        self.critic_target = Critic(self.encoder.repr_dim, kwargs['action_shape'], kwargs['feature_dim'],
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

    def update_critic(self, obs, action, reward, discount, next_obs, masked_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        masked_Q1, masked_Q2 = self.critic(masked_obs, action)
        critic_loss += 0.9 * (F.mse_loss(Q1, masked_Q1) + F.mse_loss(Q2, masked_Q2))

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics



    def update_aux(self, obs, action):
        obs_grad = compute_attribution(self.encoder, self.critic, obs, action.detach())
        mask = compute_attribution_mask(obs_grad, self.quantile)
        s_tilde = random_overlay(obs.clone())
        self.aux_optimizer.zero_grad()
        pred_attrib, aux_loss = self.compute_attribution_loss(s_tilde, action, mask)
        aux_loss.backward()
        self.aux_optimizer.step()

    def update(self, replay_iter, step):

        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float())
        aug_obs = obs.clone()
        next_obs = self.aug(next_obs.float())

        obs_grad = compute_attribution(self.encoder, self.critic, obs, action.detach())
        mask = compute_attribution_mask(obs_grad, self.quantile)
        masked_obs = obs * mask
        masked_obs[mask < 1] = random.uniform(obs.view(-1).min(), obs.view(-1).max())

        masked_obs = self.encoder(masked_obs)
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, masked_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        # this obs should be augmented obs

        self.update_aux(aug_obs, action)

        return metrics
