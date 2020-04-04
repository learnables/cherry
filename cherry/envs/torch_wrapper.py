#!/usr/bin/env python3

import numpy as np
import torch as th
import cherry as ch

from gym.spaces import Discrete

from .base import Wrapper
from .utils import is_vectorized


class Torch(Wrapper):

    """
    This wrapper converts
        * actions from Tensors to numpy,
        * states from lists/numpy to Tensors.

    Example:
        action = Categorical(Tensor([1, 2, 3])).sample()
        env.step(action)
    """

    def _convert_state(self, state):
        if isinstance(state, (float, int)):
            state = ch.totensor(state)
        if isinstance(state, dict):
            state = {k: self._convert_state(state[k]) for k in state}
        if isinstance(state, np.ndarray):
            state = ch.totensor(state)
        # we need to check for num_envs because self.is_vectorized returns
        # False when the num_envs=1, but the state still needs squeezing.
        if hasattr(self, 'num_envs') and isinstance(state, th.Tensor):
            state = state.squeeze(0)
        return state

    def _convert_atomic_action(self, action):
        if isinstance(action, th.Tensor):
            action = action.view(-1).cpu().detach().numpy()
        if self.discrete_action:
            if not isinstance(action, (int, float)):
                action = action[0]
            action = int(action)
        return action

    def _convert_action(self, action):
        if self.is_vectorized:
            if isinstance(action, th.Tensor):
                action = action.split(1, dim=0)
            elif isinstance(action, np.ndarray):
                action = action.split(1, axis=0)
            action = [self._convert_atomic_action(a) for a in action]
        else:
            action = self._convert_atomic_action(action)
        return action

    def step(self, action):
        action = self._convert_action(action)
        state, reward, done, info = self.env.step(action)
        state = self._convert_state(state)
        return state, reward, done, info

    def reset(self, *args, **kwargs):
        state = self.env.reset(*args, **kwargs)
        state = self._convert_state(state)
        return state

    def seed(self, *args, **kwargs):
        return self.env.seed(*args, **kwargs)
