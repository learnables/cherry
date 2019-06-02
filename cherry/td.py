#!/usr/bin/env python3

"""
**Description**

Utilities to implement temporal difference algorithms.
"""

import torch as th

from cherry._utils import _reshape_helper


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
    discounted = ch.rl.discount(0.99,
                                rewards,
                                dones,
                                bootstrap=1.0)
    ~~~

    """
    rewards = _reshape_helper(rewards)
    dones = _reshape_helper(dones)

    msg = 'dones and rewards must have equal length.'
    assert rewards.size(0) == dones.size(0), msg
    R = th.zeros_like(rewards[0]) + bootstrap
    discounted = th.zeros_like(rewards)
    length = discounted.size(0)
    for t in reversed(range(length)):
        if dones[t]:
            R = th.zeros_like(rewards[0])
        R = rewards[t] + gamma * R
        discounted[t] += R[0]
    return discounted


def temporal_difference(gamma, rewards, dones, values, next_values):
    """
    **Description**

    Returns the temporal difference residual.

    **Reference**

    1. Sutton, Richard S. 1988. “Learning to Predict by the Methods of Temporal Differences.” Machine Learning 3 (1): 9–44.
    2. Sutton, Richard, and Andrew Barto. 2018. Reinforcement Learning, Second Edition. The MIT Press.

    **Arguments**

    * **gamma** (float) - Discount factor.
    * **rewards** (tensor) - Tensor of rewards.
    * **dones** (tensor) - Tensor indicating episode termination.
      Entry is 1 if the transition led to a terminal (absorbing) state, 0 else.
    * **values** (tensor) - Values for the states producing the rewards.
    * **next_values** (tensor) - Values of the state obtained after the
      transition from the state used to compute the last value in `values`.

    **Example**

    ~~~python
    values = vf(replay.states())
    next_values = vf(replay.next_states())
    td_errors = temporal_difference(0.99,
                                    replay.reward(),
                                    replay.done(),
                                    values,
                                    next_values)
    ~~~
    """

    rewards = _reshape_helper(rewards)
    dones = _reshape_helper(dones)
    values = _reshape_helper(values)
    next_values = _reshape_helper(next_values)

    not_dones = 1.0 - dones
    return rewards + gamma * not_dones * next_values - values
