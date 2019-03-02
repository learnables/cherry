#!/usr/bin/env python3

import torch as th


def policy_loss(log_probs, advantages):
    """
    Advantage Actor-Critic policy loss.
    """
    msg = 'log_probs and advantages must have equal size.'
    assert log_probs.size() == advantages.size(), msg
    return -th.mean(log_probs * advantages)


def value_loss(values, rewards):
    """
    Advantage Actor-Critic value loss.
    """
    msg = 'values and rewards must have equal size.'
    assert values.size() == rewards.size(), msg
    return (rewards - values).pow(2).mean()
