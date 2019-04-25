#!/usr/bin/env python3

from gym import ActionWrapper

from .base import Wrapper


class ActionLambda(Wrapper, ActionWrapper):

    def __init__(self, env, fn):
        super(ActionLambda, self).__init__(env)
        self.fn = fn

    def action(self, action):
        return self.fn(action)
