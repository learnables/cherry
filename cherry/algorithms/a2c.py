#!/usr/bin/env python3

import torch as th


def policy_loss(log_probs, advantages):
    msg = 'log_probs and advantages must have equal size.'
    assert log_probs.size() == advantages.size(), msg
    return -th.mean(log_probs * advantages)


def value_loss(values, rewards):
    msg = 'values and rewards must have equal size.'
    assert values.size() == rewards.size(), msg
    return (rewards - values).pow(2).mean()
