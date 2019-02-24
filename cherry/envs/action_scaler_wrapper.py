#!/usr/bin/env python3

import gym
import numpy as np

from .base import Wrapper


class ActionScaler(Wrapper):

    """
    Scales the action space to be in the range (-limit, limit).

    Adapted from Vitchyr Pong's RLkit:
    https://github.com/vitchyr/rlkit/blob/master/rlkit/envs/wrappers.py#L41
    """

    def __init__(self, env, limit=1.0):
        super(ActionScaler, self).__init__(env)
        self.env = env
        self.limit = limit
        ub = np.ones(self.env.action_space.shape) * limit
        self.action_space = gym.spaces.Box(-1 * ub, ub, dtype=np.float32)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        lb = self.env.action_space.low
        ub = self.env.action_space.high
        scaled_action = lb + (action + self.limit) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)
        return self.env.step(scaled_action)
