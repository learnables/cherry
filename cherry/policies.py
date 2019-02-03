#!/usr/bin/env python3

import torch as th
import cherry as ch
import torch.nn as nn
from torch import Tensor as T
from torch.distributions import Categorical, MultivariateNormal, Normal

from gym.spaces import Discrete


class Reparameterization(object):

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

    Note: No softmax required after the linear layer of a module.
    """

    def __init__(self, env, logstd=None, use_probs=False, reparam=False):
        super(ActionDistribution, self).__init__()
        self.env = env
        if logstd is None:
            action_size = ch.utils.get_space_dimension(env.action_space)
            logstd = nn.Parameter(th.zeros(action_size))
        if isinstance(logstd, (float, int)):
            logstd = nn.Parameter(T([logstd]))
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


class CategoricalPolicy(nn.Module):

    def __init__(self, mass):
        super(CategoricalPolicy, self).__init__()
        self.mass = mass

    def forward(self, x):
        return Categorical(self.mass(x))


class DiagonalizedGaussianPolicy(nn.Module):

    def __init__(self, loc, cov=None):
        super(DiagonalizedGaussianPolicy, self).__init__()
        self.loc = loc
        if cov is None:
            cov = 1.0
        if isinstance(cov, float):
            cov = lambda x: cov
        self.cov = cov

    def forward(self, x):
        loc = self.loc(x)
        cov = self.cov(x)
        return MultivariateNormal(loc, cov)
