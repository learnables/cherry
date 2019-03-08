#!/usr/bin/env python3

import torch as th
import torch.nn as nn
import cherry as ch

from torch.distributions import (Categorical,
                                 Normal,
                                 Distribution,
                                 Bernoulli)

from gym.spaces import Discrete


class EpsilonGreedy(nn.Module):

    def __init__(self, epsilon=0.05, learnable=False):
        super(EpsilonGreedy, self).__init__()
        msg = 'epsilon is not in a valid range'
        assert epsilon >= 0.0 and epsilon <= 1.0, msg
        if learnable:
            epsilon = nn.Parameter(th.Tensor([epsilon]))
        self.epsilon = epsilon

    def forward(self, x):
        bests = x.max(dim=1, keepdim=True)[1]
        sampled = Categorical(probs=th.ones_like(x)).sample()
        probs = th.ones(x.size(0), 1) - self.epsilon
        b = Bernoulli(probs=probs).sample().long()
        ret = bests * b + (1 - b) * sampled
        return ret


class Reparameterization(object):

    """
    Reparameterized distribution.
    """

    def __init__(self, density):
        self.density = density

    def sample(self, *args, **kwargs):
        if self.density.has_rsample:
            return self.density.rsample(*args, **kwargs)
        return self.density.sample(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.density, name)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'Reparameterization(' + str(self.density) + ')'


class ActionDistribution(nn.Module):
    """
    A helper module to automatically choose the proper policy distribution,
    based on the environment action_space.
    """

    def __init__(self, env, logstd=None, use_probs=False, reparam=False):
        super(ActionDistribution, self).__init__()
        self.env = env
        if logstd is None:
            action_size = ch.envs.get_space_dimension(env.action_space)
            logstd = nn.Parameter(th.zeros(action_size))
        if isinstance(logstd, (float, int)):
            logstd = nn.Parameter(th.Tensor([logstd]))
        self.logstd = logstd
        self.use_probs = use_probs
        self.reparam = reparam
        self.is_discrete = isinstance(env.action_space, Discrete)

    def forward(self, x):
        if self.is_discrete:
            if self.use_probs:
                return Categorical(probs=x)
            return Categorical(logits=x)
        else:
            density = Normal(loc=x, scale=self.logstd.exp())
            if self.reparam:
                density = Reparameterization(density)
            return density


class TanhNormal(Distribution):

    """
    Adapted from Vitchyr Pong's RLkit:
    https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/distributions.py
    """

    def __init__(self, normal_mean, normal_std):
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)

    def sample_n(self, n):
        z = self.normal.sample_n(n)
        return th.tanh(z)

    def log_prob(self, value):
        pre_tanh_value = (th.log1p(value) - th.log1p(-value)).mul(0.5)
        offset = th.log1p(-value**2 + 1e-6)
        return self.normal.log_prob(pre_tanh_value) - offset

    def sample(self):
        z = self.normal.sample().detach()
        return th.tanh(z)

    def sample_and_log_prob(self):
        z = self.normal.sample().detach()
        value = th.tanh(z)
        offset = th.log1p(-value**2 + 1e-6)
        log_prob = self.normal.log_prob(z) - offset
        return value, log_prob

    def rsample_and_log_prob(self):
        z = self.normal.rsample()
        value = th.tanh(z)
        offset = th.log1p(-value**2 + 1e-6)
        log_prob = self.normal.log_prob(z) - offset
        return value, log_prob

    def rsample(self):
        z = self.normal.rsample()
        z.requires_grad_()
        return th.tanh(z)

