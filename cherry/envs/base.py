#!/usr/bin/env python3

import gym


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

    def __getattr__(self, attr):
        if attr in self.__dict__.keys():
            return getattr(self, attr)
        else:
            return getattr(self.env, attr)

