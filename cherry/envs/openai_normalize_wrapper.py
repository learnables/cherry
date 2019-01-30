#!/usr/bin/env python3

import numpy as np
from .base import Wrapper

from baselines.common.running_mean_std import RunningMeanStd


class OpenAINormalize(Wrapper):
    def __init__(self, env, state=True, ret=True, clipob=10.0, cliprew=10.0, gamma=0.99, eps=1e-8):
        Wrapper.__init__(self, env)
        self.env = env
        self.eps = eps
        self.gamma = gamma
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(1)
        self.state_rms = RunningMeanStd(shape=self.observation_space.shape) if state else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None

    def _obfilt(self, state):
        if self.state_rms:
            self.state_rms.update(state)
            centered = state - self.state_rms.mean
            std = np.sqrt(self.state_rms.var + self.eps)
            obs = np.clip(centered / std, -self.clipob, self.clipob)
        return obs

    def reset(self):
        self.ret = np.zeros(1)
        state = self.env.reset()
        return self._obfilt(state)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.ret = self.gamma * self.ret + reward
        state = self._obfilt(state.reshape(1, -1))[0]
        if self.ret_rms:
            reward = np.array([[reward]])
            self.ret_rms.update(self.ret)
            std = np.sqrt(self.ret_rms.var + self.eps)
            reward = np.clip(reward / std, -self.cliprew, self.cliprew)[0, 0]
        if done:
            self.ret = self.ret * 0.0
        return state, reward, done, info
