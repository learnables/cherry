#!/usr/bin/env python3

import numpy as np
import cherry as ch

from gym.spaces import Box

from .base import Wrapper

"""
Adapted from RLLab by Liyu Chen:
https://github.com/rll/rllab/blob/master/rllab/envs/normalized_env.py

The MIT License (MIT)

Copyright (c) 2016 rllab contributors

rllab uses a shared copyright model: each contributor holds copyright over
their contributions to rllab. The project versioning records all such
contribution and copyright details.
By contributing to the rllab repository through pull-request, comment,
or otherwise, the contributor releases their content to the license and
copyright terms herein.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

EPS = ch.utils.EPS
EPS = 1e-8


def update_mean(x_mean, x, weight=0.01):
    return (1 - weight) * x_mean + weight * x


def update_variance(x_var, x_mean, x, weight=0.01):
    return (1 - weight) * x_var + weight * np.square(x - x_mean)


class Normalized(Wrapper):

    """
    Normalizing wrapper for actions, states, and rewards.
    """

    def __init__(
            self,
            env,
            scale_reward=1.,
            normalize_state=True,
            normalize_reward=False,
            state_alpha=0.001,
            reward_alpha=0.001,
    ):
        super().__init__(env)
        self._scale_reward = scale_reward
        self._normalize_state = normalize_state
        self._normalize_reward = normalize_reward
        self._state_alpha = state_alpha
        flat_dim = np.prod(env.observation_space.low.shape)
        self._state_mean = np.zeros(flat_dim)
        self._state_var = np.ones(flat_dim)
        self._reward_alpha = reward_alpha
        self._reward_mean = 0.0
        self._reward_var = 1.0

    def _update_state_estimate(self, state):
        flat_state = ch.utils.flatten_state(self.env.observation_space, state)
        self._state_mean = update_mean(self._state_mean,
                                       flat_state,
                                       self._state_alpha)
        self._state_var = update_variance(self._state_var,
                                          self._state_mean,
                                          flat_state,
                                          self._state_alpha)

    def _update_reward_estimate(self, reward):
        self._reward_mean = update_mean(self._reward_mean,
                                        reward,
                                        self._reward_alpha)
        self._reward_var = update_variance(self._reward_var,
                                           self._reward_mean,
                                           reward,
                                           self._reward_alpha)

    def _apply_normalize_state(self, state):
        self._update_state_estimate(state)
        return (state - self._state_mean) / (np.sqrt(self._state_var) + EPS)

    def _apply_normalize_reward(self, reward):
        self._update_reward_estimate(reward)
        return reward / (np.sqrt(self._reward_var) + EPS)

    def reset(self):
        ret = self.env.reset()
        if self._normalize_state:
            return self._apply_normalize_state(ret)
        else:
            return ret

    def step(self, action):
        if isinstance(self.env.action_space, Box):
            lb, ub = self.env.action_space.low, self.env.action_space.high
            scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
            scaled_action = np.clip(scaled_action, lb, ub)
        else:
            scaled_action = action
        next_state, reward, done, info = self.env.step(scaled_action)
        if self._normalize_state:
            next_state = self._apply_normalize_state(next_state)
        if self._normalize_reward:
            reward = self._apply_normalize_reward(reward)
        return next_state, reward * self._scale_reward, done, info
