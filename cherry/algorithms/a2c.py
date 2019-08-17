#!/usr/bin/env python3

"""
**Description**

Helper functions for implementing A2C.

A2C simply computes the gradient of the policy as follows:

$$
\\mathbb{E} \\left[ (Q(s, a) - V(s)) \\cdot \\nabla_\\theta \\log \\pi_\\theta (a \\vert s) \\right].
$$
"""

import torch as th
from cherry import debug


def policy_loss(log_probs, advantages):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/a2c.py)

    **Description**

    The policy loss of the Advantage Actor-Critic.

    This function simply performs an element-wise multiplication and a mean reduction.

    **References**

    1. Mnih et al. 2016. “Asynchronous Methods for Deep Reinforcement Learning.” arXiv [cs.LG].

    **Arguments**

    * **log_probs** (tensor) - Log-density of the selected actions.
    * **advantages** (tensor) - Advantage of the action-state pairs.

    **Returns**

    * (tensor) - The policy loss for the given arguments.

    **Example**

    ~~~python
    advantages = replay.advantage()
    log_probs = replay.log_prob()
    loss = a2c.policy_loss(log_probs, advantages)
    ~~~
    """
    msg = 'log_probs and advantages must have equal size.'
    assert log_probs.size() == advantages.size(), msg
    if debug.IS_DEBUGGING:
        if advantages.requires_grad:
            debug.logger.warning('A2C:policy_loss: advantages.requires_grad is True.')
        if not log_probs.requires_grad:
            debug.logger.warning('A2C:policy_loss: log_probs.requires_grad is False.')
    return -th.mean(log_probs * advantages)


def state_value_loss(values, rewards):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/a2c.py)

    **Description**

    The state-value loss of the Advantage Actor-Critic.

    This function is equivalent to a MSELoss.

    **References**

    1. Mnih et al. 2016. “Asynchronous Methods for Deep Reinforcement Learning.” arXiv [cs.LG].

    **Arguments**

    * **values** (tensor) - Predicted values for some states.
    * **rewards** (tensor) - Observed rewards for those states.

    **Returns**

    * (tensor) - The value loss for the given arguments.

    **Example**

    ~~~python
    values = replay.value()
    rewards = replay.reward()
    loss = a2c.state_value_loss(values, rewards)
    ~~~
    """
    msg = 'values and rewards must have equal size.'
    assert values.size() == rewards.size(), msg
    if debug.IS_DEBUGGING:
        if rewards.requires_grad:
            debug.logger.warning('A2C:state_value_loss: rewards.requires_grad is True.')
        if not values.requires_grad:
            debug.logger.warning('A2C:state_value_loss: values.requires_grad is False.')
    return (rewards - values).pow(2).mean()
