#!/usr/bin/env python3

import gym

from .utils import get_space_dimension


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
    def state_size(self):
        return get_space_dimension(self.observation_space)

    @property
    def action_size(self):
        return get_space_dimension(self.action_space)

    def __getattr__(self, attr):
        if attr in self.__dict__.keys():
            return getattr(self, attr)
        else:
            return getattr(self.env, attr)
