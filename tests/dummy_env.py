#!/usr/bin/env python3

import random
import gym
import numpy as np


class Dummy(gym.Env):

    """
    A dummy environment that returns random states and rewards.
    """

    def __init__(self):
        low = np.array([-5, -5, -5, -5, -5])
        high = -np.array([-5, -5, -5, -5, -5])
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.rng = random.Random()

    def step(self, action):
        assert self.observation_space.contains(action)
        next_state = self.observation_space.sample()
        reward = action.sum()
        done = random.random() > 0.95
        info = {}
        return next_state, reward, done, info

    def reset(self):
        return self.observation_space.sample()

    def seed(self, seed=1234):
        self.rng.seed(seed)
        np.random.seed(seed)

    def _render(self, mode='human', close=False):
        pass

    def _take_action(self, action):
        pass

    def _get_reward(self):
        return self.rng.randint(0, 10)
