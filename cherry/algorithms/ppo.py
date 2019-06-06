#!/usr/bin/env python3

"""
**Description**

Helper functions for implementing PPO.
"""

import torch as th


def policy_loss(new_log_probs, old_log_probs, advantages, clip=0.1):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/ppo.py)

    **Description**

    The clipped policy loss of Proximal Policy Optimization.

    **References**

    1. Schulman et al. 2017. “Proximal Policy Optimization Algorithms.” arXiv [cs.LG].

    **Arguments**

    * **new_log_probs** (tensor) - The log-density of actions from the target policy.
    * **old_log_probs** (tensor) - The log-density of actions from the behaviour policy.
    * **advantages** (tensor) - Advantage of the actions.
    * **clip** (float, *optional*, default=0.1) - The clipping coefficient.

    **Returns**

    * (tensor) - The clipped policy loss for the given arguments.

    **Example**

    ~~~python
    advantage = ch.pg.generalized_advantage(GAMMA,
                                            TAU,
                                            replay.reward(),
                                            replay.done(),
                                            replay.value(),
                                            next_state_value)
    new_densities = policy(replay.state())
    new_logprobs = new_densities.log_prob(replay.action())
    loss = policy_loss(new_logprobs,
                       replay.logprob().detach(),
                       advantage.detach(),
                       clip=0.2)
    ~~~
    """
    msg = 'new_log_probs, old_log_probs and advantages must have equal size.'
    assert new_log_probs.size() == old_log_probs.size() == advantages.size(),\
        msg
    ratios = th.exp(new_log_probs - old_log_probs)
    obj = ratios * advantages
    obj_clip = ratios.clamp(1.0 - clip, 1.0 + clip) * advantages
    return - th.min(obj, obj_clip).mean()


def state_value_loss(new_values, old_values, rewards, clip=0.1):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/ppo.py)

    **Description**

    The clipped state-value loss of Proximal Policy Optimization.

    **References**

    1. PPO paper

    **Arguments**

    * **new_values** (tensor) - State values from the optimized value function.
    * **old_values** (tensor) - State values from the reference value function.
    * **rewards** (tensor) -  Observed rewards.
    * **clip** (float, *optional*, default=0.1) - The clipping coefficient.

    **Returns**

    * (tensor) - The clipped value loss for the given arguments.

    **Example**

    ~~~python
    values = v_function(batch.state())
    value_loss = ppo.state_value_loss(values,
                                      batch.value().detach(),
                                      batch.reward(),
                                      clip=0.2)
    ~~~
    """
    msg = 'new_values, old_values, and rewards must have equal size.'
    assert new_values.size() == old_values.size() == rewards.size(), msg
    loss = (rewards - new_values)**2
    clipped_values = old_values + (new_values - old_values).clamp(-clip, clip)
    clipped_loss = (rewards - clipped_values)**2
    return 0.5 * th.max(loss, clipped_loss).mean()
