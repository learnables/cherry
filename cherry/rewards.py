#!/usr/bin/env python3


"""
[[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/rewards.py)

**Description**

Utilities to manipulate rewards, such as discounting or advantage computation.
"""

import torch as th


def discount(gamma, rewards, dones, bootstrap=0.0):
    """
    **Description**

    Discounts rewards at an rate of gamma.

    **References**

    1. Sutton, Richard, and Andrew Barto. 2018. Reinforcement Learning, Second Edition. The MIT Press.

    **Arguments**

    * **gamma** (float) - Discount factor.
    * **rewards** (tensor) - Tensor of rewards.
    * **dones** (tensor) - Tensor indicating episode termination.
      Entry is 1 if the transition led to a terminal (absorbing) state, 0 else.
    * **bootstrap** (float, *optional*, default=0.0) - Bootstrap the last
      reward with this value.

    **Returns**

    * tensor - Tensor of discounted rewards.

    **Example**

    ~~~python
    rewards = th.ones(23, 1) * 8
    dones = th.zeros_like(rewards)
    dones[-1] += 1.0
    discounted = ch.rewards.discount(0.99,
                                     rewards,
                                     dones,
                                     bootstrap=1.0)
    ~~~

    """
    if len(rewards.size()) == 1:
        rewards = rewards.view(-1, 1)
    if len(dones.size()) == 1:
        dones = dones.view(-1, 1)
    R = bootstrap
    discounted = th.zeros_like(rewards)
    length = discounted.size(0)
    for t in reversed(range(length)):
        if dones[t]:
            R = 0.0
        R = rewards[t] + gamma * R
        discounted[t] += R
    return discounted


def generalized_advantage():
    pass


def generalized_advantage_estimate(gamma, tau, rewards, dones, values, next_value):
    """
    """
    msg = 'GAE needs as many rewards, values and dones.'
    assert len(values) == len(rewards) == len(dones), msg
    advantages = []
    advantage = 0
    for i in reversed(range(len(rewards))):
        td_error = rewards[i] + (1.0 - dones[i]) * gamma * next_value - values[i]
        advantage = advantage * tau * gamma * (1.0 - dones[i]) + td_error
        advantages.insert(0, advantage)
        next_value = values[i]
    return advantages
