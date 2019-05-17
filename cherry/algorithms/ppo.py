#!/usr/bin/env python3

import torch as th


def policy_loss(new_log_probs, old_log_probs, advantages, clip=0.1):
    """

    **Description**
    
    To calculate the clipped policy loss from old policy to new policy.

    **Arguments**
    
    * **new_log_probs** (tensor) - The log density of the actions from the new policy on some states
    * **old_log_probs** (tensor) - The log density of the actions from the old policy on some states
    * **advantages** (tensor) - The advantage of a state.
    * **clip** (float) - The hyperparameter saying how far away the new policy is allowed to go from the old.

    **References**

    **Example**
    ~~~python
    advantages = replay.advantages
    log_probs = replay.log_probs
    loss = ppo.policy_loss(log_probs, advantages)
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
    **Description**
    
    To calculate the clipped value loss of some states from old policy to new policy.

    **Arguments**

    * **new_values** (tensor) - The state's V value resulted from new policy.
    * **old_values** (tensor) - The state's V value resulted from old policy.
    * **rewards** (tensor) - Observerd rewards during the transition.
    * **clip** (float) - The hyperparameter saying how far away the new value is allowed to go from the old.

    **References**

    **Example**
    ~~~python
    batch = replay.sample(PPO_BSZ)
    values = policy(batch.state())
    value_loss = ppo.state_value_loss(values,
                                      batch.value().detach(),
                                      batch.reward(),
                                      clip=0.1)
    ~~~
    
    """
    msg = 'new_values, old_values, and rewards must have equal size.'
    assert new_values.size() == old_values.size() == rewards.size(), msg
    loss = (rewards - new_values)**2
    clipped_values = old_values + (new_values - old_values).clamp(-clip, clip)
    clipped_loss = (rewards - clipped_values)**2
    return 0.5 * th.max(loss, clipped_loss).mean()
