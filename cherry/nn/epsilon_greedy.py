#!/usr/bin/env python3

import torch as th
from torch import nn
from torch.distributions import Bernoulli, Categorical


class EpsilonGreedy(nn.Module):

    """
    [[Source]]()

    **Description**

    Epsilon greedy action selection.

    **References**

    **Arguments**

    **Returns**

    **Example**

    """

    def __init__(self, epsilon=0.05, learnable=False):
        super(EpsilonGreedy, self).__init__()
        msg = 'EpsilonGreedy: epsilon is not in a valid range.'
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
