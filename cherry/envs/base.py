#!/usr/bin/env python3

import operator
from functools import reduce

import gym
from gym.spaces import Discrete, Box


def get_space_dimension(space):
    msg = 'Space type not supported.'
    assert isinstance(space, (Box, Discrete)), msg
    if isinstance(space, Discrete):
        return space.n
    if isinstance(space, Box):
        return reduce(operator.mul, space.shape, 1)


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

