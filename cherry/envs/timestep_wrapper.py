#!/usr/bin/env python3

import numpy as np

from gym import ObservationWrapper
from gym.spaces import Box

from .base import Wrapper


class AddTimestep(Wrapper, ObservationWrapper):

    """
    Adds a timestep information to the state input.

    Modified from Ilya Kostrikov's implementation:

    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/
    """

    def __init__(self, env=None):
        super(AddTimestep, self).__init__(env)
        self.observation_space = Box(
            low=self.observation_space.low[0],
            high=self.observation_space.high[0],
            shape=[self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return np.concatenate((observation.reshape(-1),
                               [self.env._elapsed_steps]))
