#!/usr/bin/env python3

import numpy as np
import cv2

from gym.spaces import Box

from .base import Wrapper

cv2.ocl.setUseOpenCL(False)

"""
Inspired from OpenAI's baselines:

https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

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


class Atari(Wrapper):

    """
    Process Atari frames similarly to what DeepMind does.

    * Skip 3 out of 4 frames,
    * Stack the last 4 frames,
    * Warp state to a 84x84 image (optionally grayscale).

    Note: The reward clipping is available in cherry.envs.ClipRewards.
    """

    def __init__(self, env, grayscale=True, skip=4, stack=4, warp=84):
        super(Atari, self).__init__(env)
        self.grayscale = grayscale
        self.warp = warp
        self.skip_frames = skip
        self.stack_frames = stack
        num_channels = 1 if grayscale else 3
        self.observation_space = Box(low=0,
                                     high=255,
                                     shape=(stack, warp, warp, num_channels),
                                     dtype=env.observation_space.dtype)
        self._state_history = []

    def _preprocess_frame(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame,
                           (self.warp, self.warp),
                           interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame

    def _stack_states(self):
        self._state_history = self._state_history[-self.stack_frames:]
        return np.stack(self._state_history).transpose(3, 0, 1, 2)

    def reset(self, *args, **kwargs):
        state = self.env.reset(*args, **kwargs)
        state = self._preprocess_frame(state)
        self._state_history = [state, ] * self.stack_frames
        state = self._stack_states()
        return state

    def step(self, *args, **kwargs):
        """Sum rewards, max over observations."""
        total_reward = 0.0
        max_state = self.env.observation_space.low.copy()
        for _ in range(self.skip_frames):
            state, reward, done, info = self.env.step(*args, **kwargs)
            total_reward += reward
            max_state = np.maximum(state, max_state)

        state = self._preprocess_frame(max_state)
        self._state_history.append(state)
        state = self._stack_states()
        return state, total_reward, done, info
