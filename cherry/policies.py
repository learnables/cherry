#!/usr/bin/env python3

import torch.nn as nn
from torch import Tensor as T
from torch.distributions import Categorical, MultivariateNormal, Normal

from gym.spaces import Discrete


class ActionDistribution(nn.Module):
    """
    A helper module to automatically choose the proper policy distribution,
    based on the environment action_space.

    Note: No softmax required after the linear layer of a module.
    """

    def __init__(self, env, cov=1e-2):
        super(ActionDistribution, self).__init__()
        self.env = env
        if isinstance(cov, (float, int)):
            cov = nn.Parameter(T([cov]))
        self.cov = cov
        self.is_discrete = isinstance(env.action_space, Discrete)

    def forward(self, x):
        if self.is_discrete:
            return Categorical(logits=x)
        else:
            return Normal(x, self.cov)


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
