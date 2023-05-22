import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from drq import DrQAgent
from drqv2 import DrQV2Agent, RandomShiftsAug, Actor, Critic

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



class CURLAgent(DrQV2Agent):
    def __init__(self, aux_lr=0.3, aux_beta=0.9, **kwargs):
        super().__init__(**kwargs)
        self.encoder = CNNEncoder(kwargs['obs_shape']).to(self.device)
        self.curl_head = CURLHead(self.encoder.repr_dim).to(self.device)

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
        self.curl_optimizer = torch.optim.Adam(
            self.curl_head.parameters(), lr=aux_lr, betas=(aux_beta, 0.999)
        )

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



    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        original_obs = obs.clone()
        pos_obs = obs.clone()
        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        original_obs = self.aug(original_obs.float())
        pos_obs = self.aug(pos_obs.float())
        # encode
        obs = self.encoder(obs)

        with torch.no_grad():
            next_obs = self.encoder(next_obs)


        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update curl
        obs = self.encoder(original_obs)
        with torch.no_grad():
            pos = self.encoder(pos_obs)
        metrics.update(self.update_curl(obs, pos))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

