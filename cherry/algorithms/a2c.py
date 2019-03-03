#!/usr/bin/env python3

"""
Helper functions for A2C.
"""

import torch as th


def policy_loss(log_probs, advantages):
    """
    [[Source]]()

    **Description**

    Advantage Actor-Critic policy loss.

    **References**

    **Arguments**

    **Returns**

    **Example**

    ~~~python
    advantages = replay.advantages
    log_probs = replay.log_probs
    loss = a2c.policy_loss(log_probs, advantages)
    ~~~
    """
    msg = 'log_probs and advantages must have equal size.'
    assert log_probs.size() == advantages.size(), msg
    return -th.mean(log_probs * advantages)


def state_value_loss(values, rewards):
    """
    Advantage Actor-Critic value loss.
    """
    msg = 'values and rewards must have equal size.'
    assert values.size() == rewards.size(), msg
    return (rewards - values).pow(2).mean()
