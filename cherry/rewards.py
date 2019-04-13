#!/usr/bin/env python3


"""
[[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/rewards.py)

**Description**

Utilities to manipulate rewards, such as discounting or advantage computation.
"""

import torch as th


def _reshape_helper(tensor):
    if len(tensor.size()) == 1:
        return tensor.view(-1, 1)
    return tensor


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


def generalized_advantage(gamma,
                          tau,
                          rewards,
                          dones,
                          values,
                          next_value):
    """
    **Description**

    Computes the generalized advantage estimator. (GAE)

    **References**

    1. Schulman et al. 2015. “High-Dimensional Continuous Control Using Generalized Advantage Estimation”
    2. https://github.com/joschu/modular_rl/blob/master/modular_rl/core.py#L49

    **Arguments**

    * **gamma** (float) - Discount factor.
    * **tau** (float) - Bias-variance trade-off.
    * **rewards** (tensor) - Tensor of rewards.
    * **dones** (tensor) - Tensor indicating episode termination.
      Entry is 1 if the transition led to a terminal (absorbing) state, 0 else.
    * **values** (tensor) - Values for the states producing the rewards.
    * **next_value** (tensor) - Value of the state obtained after the
      transition from the state used to compute the last value in `values`.

    **Returns**

    * tensor - Tensor of advantages.

    **Example**
    ~~~python
    mass, next_value = policy(replay.next_states[-1])
    advantages = generalized_advantage(0.99,
                                       0.95,
                                       replay.)
    ~~~
    """

    rewards = _reshape_helper(rewards)
    dones = _reshape_helper(dones)
    values = _reshape_helper(values)
    next_value = _reshape_helper(next_value)

    msg = 'rewards, values, and dones must have equal length.'
    assert len(values) == len(rewards) == len(dones), msg

    td_errors = temporal_difference(gamma, rewards, dones, values, next_value)
    advantages = discount(tau * gamma, td_errors, dones)
    return advantages


def temporal_difference(gamma, rewards, dones, values, next_value):
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
    * **next_value** (tensor) - Value of the state obtained after the
      transition from the state used to compute the last value in `values`.

    **Example**

    ~~~python
    value = vf(state)
    td_errors = temporal_difference(0.99,
                                    replay.rewards,
                                    replay.dones,
                                    replay.values,
                                    next_value)
    ~~~
    """

    rewards = _reshape_helper(rewards)
    dones = _reshape_helper(dones)
    values = _reshape_helper(values)
    next_value = _reshape_helper(next_value)

    all_values = th.cat((values, next_value), dim=0)
    not_dones = 1.0 - dones
    return rewards + gamma * not_dones * all_values[1:] - all_values[:-1]
