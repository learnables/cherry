#!/usr/bin/env python3

import torch as th
from torch.distributions import Distribution, Normal


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
        offset = th.log1p(-value**2)
        return self.normal.log_prob(pre_tanh_value) - offset

    def sample(self):
        z = self.normal.sample().detach()
        return th.tanh(z)

    def rsample(self):
        z = self.normal.rsample()
        z.requires_grad_()
        return th.tanh(z)

