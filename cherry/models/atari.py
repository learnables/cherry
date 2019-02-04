#!/usr/bin/env python3

import torch.nn as nn
from cherry.nn.init import atari_init_

"""
This model is largely inspired from Ilya Kostrikov's implementation:

https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/
"""


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NatureFeatures(nn.Sequential):

    def __init__(self, input_size=4, hidden_size=512):
        super(NatureFeatures, self).__init__(
            atari_init_(nn.Conv2d(input_size, 32, 8, stride=4)),
            nn.ReLU(),
            atari_init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            atari_init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            atari_init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )


class NatureActor(nn.Linear):

    def __init__(self, input_size, output_size):
        super(NatureActor, self).__init__(input_size, output_size)
        atari_init_(self, gain=0.01)


class NatureCritic(nn.Linear):

    def __init__(self, input_size, output_size=1):
        super(NatureCritic, self).__init__(input_size, output_size)
        atari_init_(self, gain=1.0)
