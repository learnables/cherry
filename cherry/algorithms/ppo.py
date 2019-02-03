#!/usr/bin/env python3

import torch as th


def policy_loss(new_log_probs, old_log_probs, advantages, clip=0.1):
    ratios = th.exp(new_log_probs - old_log_probs)
    obj = ratios * advantages
    obj_clip = ratios.clamp(1.0 - clip, 1.0 + clip) * advantages
    return - th.min(obj, obj_clip).mean()


def value_loss(new_values, old_values, rewards, clip=0.1):
    loss = (rewards - new_values).pow(2).mul(0.5).mean()
    clipped_values = old_values + (new_values - old_values).clamp(-clip, clip)
    clipped_loss = (rewards - clipped_values).pow(2).mul(0.5).mean()
    return th.max(loss, clipped_loss)
