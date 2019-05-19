#!/usr/bin/env python3

import numpy as np
import torch as th
import cherry as ch

from gym.spaces import Discrete

from .base import Wrapper


class Torch(Wrapper):

    """
    This wrapper converts
        * actions from Tensors to numpy,
        * states from lists/numpy to Tensors.

    Example:
        action = Categorical(Tensor([1, 2, 3])).sample()
        env.step(action)
    """

    def __init__(self, env):
        super(Torch, self).__init__(env)

    def _convert_state(self, state):
        if isinstance(state, (float, int)):
            state = th.Tensor([state])
        if isinstance(state, dict):
            state = {k: self._convert_state(state[k]) for k in state}
            return state
        if isinstance(state, np.ndarray):
            return ch.totensor(state)
        return state

    def step(self, action):
        if isinstance(action, th.Tensor):
            action = action.view(-1).data.numpy()
        if isinstance(self.env.action_space, Discrete):
            action = action[0]
        state, reward, done, info = self.env.step(action)
        state = self._convert_state(state)
        return state, reward, done, info

    def reset(self, *args, **kwargs):
        state = self.env.reset(*args, **kwargs)
        state = self._convert_state(state)
        return state

    def seed(self, *args, **kwargs):
        return self.env.seed(*args, **kwargs)
