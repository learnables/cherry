#!/usr/bin/env python3

import numpy as np
from .base_wrapper import Wrapper


class StateNormalizer(Wrapper):

    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/envs/normalizer_wrapper.py)

    **Description**

    Normalizes the states with a running average.

    **Arguments**

     * **env** (Environment) - Environment to normalize.
     * **statistics** (dict, *optional*, default=None) - Dictionary used to
        bootstrap the normalizing statistics.
     * **beta** (float, *optional*, default=0.99) - Moving average weigth.
     * **eps** (float, *optional*, default=1e-8) - Numerical stability.

    **Credit**

    Adapted from Tristan Deleu's implementation.

    **Example**
    ~~~python
    env = gym.make('CartPole-v0')
    env = cherry.envs.StateNormalizer(env)
    env2 = gym.make('CartPole-v0')
    env2 = cherry.envs.StateNormalizer(env2,
                                       statistics=env.statistics)
    ~~~
    """

    def __init__(self, env, statistics=None, beta=0.99, eps=1e-8):
        super(StateNormalizer, self).__init__(env)
        self.beta = beta
        self.eps = eps
        if statistics is not None and 'mean' in statistics:
            self._state_mean = np.copy(statistics['mean'])
        else:
            self._state_mean = np.zeros(self.observation_space.shape)

        if statistics is not None and 'var' in statistics:
            self._state_var = np.copy(statistics['var'])
        else:
            self._state_var = np.ones(self.observation_space.shape)

    @property
    def statistics(self):
        return {
            'mean': self._state_mean,
            'var': self._state_var,
        }

    def _state_normalize(self, state):
        self._state_mean = self.beta * self._state_mean + (1.0 - self.beta) * state
        self._state_var = self.beta * self._state_var + (1.0 - self.beta) * np.square(state - self._state_mean)
        return (state - self._state_mean) / (np.sqrt(self._state_var) + self.eps)

    def reset(self, *args, **kwargs):
        state = self.env.reset(*args, **kwargs)
        return self._state_normalize(state)

    def step(self, *args, **kwargs):
        state, reward, done, infos = self.env.step(*args, **kwargs)
        return self._state_normalize(state), reward, done, infos
