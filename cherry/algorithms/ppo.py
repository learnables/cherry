#!/usr/bin/env python3

import torch as th


def policy_loss(new_log_probs, old_log_probs, advantages, clip=0.1):
    """
    Clipped policy loss.

    Arguments:

    * new_log_probs: (tensor)
    * old_log_probs: (tensor)
    * advantages: (tensor)
    * clip: (tensor)
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
    Clipped value loss.

    Arguments:

    * new_values: (tensor)
    * old_values: (tensor)
    * rewards: (tensor)
    * clip: (tensor)
    """
    msg = 'new_values, old_values, and rewards must have equal size.'
    assert new_values.size() == old_values.size() == rewards.size(), msg
    loss = (rewards - new_values)**2
    clipped_values = old_values + (new_values - old_values).clamp(-clip, clip)
    clipped_loss = (rewards - clipped_values)**2
    return 0.5 * th.max(loss, clipped_loss).mean()
