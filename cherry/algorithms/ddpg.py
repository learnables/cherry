#!/usr/bin/env python3

from torch.nn import functional as F
from cherry import debug


def state_value_loss(values, next_values, rewards, dones, gamma):
    """
    Arguments:

    """
    msg = 'rewards, values, and next_values must have equal size.'
    assert values.size() == next_values.size() == rewards.size(), msg
    if debug.IS_DEBUGGING:
        if rewards.requires_grad:
            debug.logger.warning('DDPG:state_value_loss: rewards.requires_grad is True.')
        if next_values.requires_grad:
            debug.logger.warning('DDPG:state_value_loss: next_values.requires_grad is True.')
        if not values.requires_grad:
            debug.logger.warning('DDPG:state_value_loss: values.requires_grad is False.')
    v_target = rewards + (1.0 - dones) * gamma * next_values
    return F.mse_loss(values, v_target)
