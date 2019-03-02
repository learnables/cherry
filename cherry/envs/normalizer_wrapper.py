#!/usr/bin/env python3

import numpy as np
from .base import Wrapper

"""
Code adapted from OpenAI Baslines, under the following license.

The MIT License

Copyright (c) 2017 OpenAI (http://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
                                                self.mean,
                                                self.var,
                                                self.count,
                                                batch_mean,
                                                batch_var,
                                                batch_count)


def update_mean_var_count_from_moments(mean,
                                       var,
                                       count,
                                       batch_mean,
                                       batch_var,
                                       batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class Normalizer(Wrapper):

    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/envs/normalizer_wrapper.py)

    **Description**

    Normalizes the states and rewards with a running average.

    **Arguments**

     * **env** (Environment) - Environment to normalize.
     * **states** (bool, *optional*, default=True) - Whether to normalize the
       states.
     * **rewards** (bool, *optional*, default=True) - Whether to normalize the
       rewards.
     * **clip_states** (bool, *optional*, default=10.0) - Clip each state
       dimension between [-clip_states, clip_states].
     * **clip_rewards** (float, *optional*, default=10.0) - Clip rewards
       between [-clip_rewards, clip_rewards].
     * **gamma** (float, *optional*, default=0.99) - Discount factor for
       rewards running averages.
     * **eps** (float, *optional*, default=1e-8) - Numerical stability.

    **Credit**

    Adapted from OpenAI's baselines implementation.

    **Example**
    ~~~python
    env = gym.make('CartPole-v0')
    env = cherry.envs.Normalizer(env,
                                 states=True,
                                 rewards=False)
    ~~~
    """

    def __init__(self,
                 env,
                 states=True,
                 rewards=True,
                 clip_states=10.0,
                 clip_rewards=10.0,
                 gamma=0.99,
                 eps=1e-8):
        Wrapper.__init__(self, env)
        self.env = env
        self.eps = eps
        self.gamma = gamma
        self.clipob = clip_states
        self.cliprew = clip_rewards
        self.ret = np.zeros(1)
        if states:
            self.state_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.ret_rms = RunningMeanStd(shape=()) if rewards else None

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
        state = self._obfilt(state)
        if self.ret_rms:
            reward = np.array([[reward]])
            self.ret_rms.update(self.ret)
            std = np.sqrt(self.ret_rms.var + self.eps)
            reward = np.clip(reward / std, -self.cliprew, self.cliprew)[0, 0]
        if done:
            self.ret = self.ret * 0.0
        return state, reward, done, info
