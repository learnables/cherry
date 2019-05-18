#!/usr/bin/env python3

"""
Helper functions for A2C.
"""

import torch as th


def policy_loss(log_probs, advantages):
    """
    [[Source]]()

    **Description**

    To calculate the policy loss from old policy to new policy in A2C method.

    **Arguments**
    
    * **log_probs** (tensor) - The log density of the actions from the new policy on some states
    * **advantages** (tensor) - The advantage of a state.

    **References**

    **Returns**

    (tensor)

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
    **Description**

    To calculate the value loss of some states from old policy to new policy in A2C method.
    
    **Arguments**
    
    * **values** (tensor) - The state's V value.
    * **rewards** (tensor) - Observerd rewards during the transition.

    **References**


    **Example**
    ~~~python
    rewards = ch.rewards.discount(GAMMA,
                                  replay.reward(),
                                  replay.done(),
                                  bootstrap=next_state_value)
    rewards = rewards.detach()
    value_loss = a2c.state_value_loss(replay.value(), rewards)
    ~~~
    """
    msg = 'values and rewards must have equal size.'
    assert values.size() == rewards.size(), msg
    return (rewards - values).pow(2).mean()
