#!/usr/bin/env python3

from gym import ObservationWrapper

from .base_wrapper import Wrapper


class StateLambda(Wrapper, ObservationWrapper):

    def __init__(self, env, fn):
        super(StateLambda, self).__init__(env)
        self.fn = fn

    def observation(self, observation):
        return self.fn(observation)
