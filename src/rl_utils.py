import torch
from torchvision.utils import make_grid

from captum.attr import GuidedBackprop, GuidedGradCam


class HookFeatures:
    def __init__(self, module):
        self.feature_hook = module.register_forward_hook(self.feature_hook_fn)

    def feature_hook_fn(self, module, input, output):
        self.features = output.clone().detach()
        self.gradient_hook = output.register_hook(self.gradient_hook_fn)

    def gradient_hook_fn(self, grad):
        self.gradients = grad

    def close(self):
        self.feature_hook.remove()
        self.gradient_hook.remove()


class ModelWrapper(torch.nn.Module):
    def __init__(self, encoder, critic, obs_sensor, action=None):
        super(ModelWrapper, self).__init__()
        self.encoder = encoder
        self.critic = critic
        self.action = action
        self.obs_sensor = obs_sensor

    def forward(self, obs):
        obs = self.encoder(obs)
        obs_combined = torch.cat([obs, self.obs_sensor], dim=1)
        if self.action is None:
            return self.critic(obs_combined)[0]
        return self.critic(obs_combined, self.action)[0]


def compute_guided_backprop(obs, obs_sensor, action, encoder, critic):
    model = ModelWrapper(encoder, critic, obs_sensor, action=action)
    gbp = GuidedBackprop(model)
    attribution = gbp.attribute(obs)
    return attribution

def compute_guided_gradcam(obs, obs_sensor, action, encoder, critic):
    obs.requires_grad_()
    obs.retain_grad()
    model = ModelWrapper(encoder, critic, obs_sensor, action=action)
    gbp = GuidedGradCam(model,layer=model.model.encoder.head_cnn.layers)
    attribution = gbp.attribute(obs,attribute_to_layer_input=True)
    return attribution

def compute_vanilla_grad(critic_target, obs, action):
    obs.requires_grad_()
    obs.retain_grad()
    q, q2 = critic_target(obs, action.detach())
    q.sum().backward()
    return obs.grad


def compute_attribution(encoder, critic, obs, obs_sensor, action=None,method="guided_backprop"):
    if method == "guided_backprop":
        return compute_guided_backprop(obs, obs_sensor, action, encoder, critic)
    if method == 'guided_gradcam':
        return compute_guided_gradcam(obs,action, obencoder, critic)


def compute_features_attribution(critic_target, obs, action):
    obs.requires_grad_()
    obs.retain_grad()
    hook = HookFeatures(critic_target.encoder)
    q, _ = critic_target(obs, action.detach())
    q.sum().backward()
    features_gardients = hook.gradients
    hook.close()
    return obs.grad, features_gardients


def compute_attribution_mask(obs_grad, quantile=0.95):
    mask = []
    for i in [0, 3, 6]:
        attributions = obs_grad[:, i : i + 3].abs().max(dim=1)[0]
        q = torch.quantile(attributions.flatten(1), quantile, 1)
        mask.append((attributions >= q[:, None, None]).unsqueeze(1).repeat(1, 3, 1, 1))
    return torch.cat(mask, dim=1)


def make_obs_grid(obs, n=4):
    sample = []
    for i in range(n):
        for j in range(0, 9, 3):
            sample.append(obs[i, j : j + 3].unsqueeze(0))
    sample = torch.cat(sample, 0)
    return make_grid(sample, nrow=3) / 255.0


def make_attribution_pred_grid(attribution_pred, n=4):
    return make_grid(attribution_pred[:n], nrow=1)


def make_obs_grad_grid(obs_grad, n=4):
    sample = []
    for i in range(n):
        for j in range(0, 9, 3):
            channel_attribution, _ = torch.max(obs_grad[i, j : j + 3], dim=0)
            sample.append(channel_attribution[(None,) * 2] / channel_attribution.max())
    sample = torch.cat(sample, 0)
    q = torch.quantile(sample.flatten(1), 0.97, 1)
    sample[sample <= q[:, None, None, None]] = 0
    return make_grid(sample, nrow=3)
