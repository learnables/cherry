#!/usr/bin/env python3

import gym

from .utils import get_space_dimension, is_vectorized, is_discrete


class Wrapper(gym.Wrapper):

    """
    This class allows to chain Environment Wrappers while still being able to
    access the properties of wrapped wrappers.

    Example:

        env = gym.make('MyEnv-v0')
        env = envs.Logger(env)
        env = envs.Runner(env)
        env.log('asdf', 23)  # Uses log() method from envs.Logger.
    """

    @property
    def discrete_action(self):
        return is_discrete(self.action_space)

    @property
    def discrete_state(self):
        return is_discrete(self.observation_space)

    @property
    def state_size(self):
        vectorized = not is_vectorized(self)
        return get_space_dimension(self.observation_space, vectorized=vectorized)

    @property
    def action_size(self):
        vectorized = not is_vectorized(self)
        return get_space_dimension(self.action_space, vectorized=vectorized)

    def __getattr__(self, attr):
        if attr in self.__dict__.keys():
            return getattr(self, attr)
        else:
            return getattr(self.env, attr)
