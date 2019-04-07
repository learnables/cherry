#!/usr/bin/env python3

from torch.nn import functional as F


def state_value_loss(values, next_values, rewards, dones, gamma):
    """
    Arguments:

    """
    msg = 'rewards, values, and next_values must have equal size.'
    assert values.size() == next_values.size() == rewards.size(), msg
    v_target = rewards + (1.0 - dones) * gamma * next_values
    return F.mse_loss(values, v_target.detach())
