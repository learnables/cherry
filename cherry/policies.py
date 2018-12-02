#!/usr/bin/env python3

import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal


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
