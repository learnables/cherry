#!/usr/bin/env python3

import torch as th
from torch import nn
from torch.distributions import Bernoulli, Categorical


class EpsilonGreedy(nn.Module):

    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/nn/epsilon_greedy.py)

    **Description**

    Samples actions from a uniform distribution with probability `epsilon` or
    the one maximizing the input with probability `1 - epsilon`.

    **References**

    1. Sutton, Richard, and Andrew Barto. 2018. Reinforcement Learning, Second Edition. The MIT Press.

    **Arguments**

    * **epsilon** (float, *optional*, default=0.05) - The epsilon factor.
    * **learnable** (bool, *optional*, default=False) - Whether the epsilon
    factor is a learnable parameter or not.

    **Example**

    ~~~python
    egreedy = EpsilonGreedy()
    q_values = q_value(state)  # NxM tensor
    actions = egreedy(q_values)  # Nx1 tensor of longs
    ~~~

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
