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

    """
    <a href="https://github.com/seba-1511/cherry/blob/master/cherry/models/atari.py" class="source-link">[Source]</a>

    ## Description

    The convolutional body of the DQN architecture.

    ## References

    1. Mnih et al. 2015. “Human-Level Control through Deep Reinforcement Learning.”
    2. Mnih et al. 2016. “Asynchronous Methods for Deep Reinforcement Learning.”

    ## Credit

    Adapted from Ilya Kostrikov's implementation.

    """

    def __init__(self, input_size=4, output_size=512, hidden_size=64 * 7 * 7):
        """
        ## Arguments

        * `input_size` (int) - Number of channels.
          (Stacked frames in original implementation.)
        * `output_size` (int, *optional*, default=512) - Size of the output
          representation.
        * `hidden_size` (int, *optional*, default=1568) - Size of the representation
          after the convolutional layers
        """
        super(NatureFeatures, self).__init__(
            atari_init_(nn.Conv2d(input_size, 32, 8, stride=4, padding=0)),
            nn.ReLU(),
            atari_init_(nn.Conv2d(32, 64, 4, stride=2, padding=0)),
            nn.ReLU(),
            atari_init_(nn.Conv2d(64, 64, 3, stride=1, padding=0)),
            nn.ReLU(),
            Flatten(),
            atari_init_(nn.Linear(hidden_size, output_size)),
            nn.ReLU()
        )


class NatureActor(nn.Linear):

    """
    <a href="https://github.com/seba-1511/cherry/blob/master/cherry/models/atari.py" class="source-link">[Source]</a>

    ## Description

    The actor head of the A3C architecture.

    ## References

    1. Mnih et al. 2015. “Human-Level Control through Deep Reinforcement Learning.”
    2. Mnih et al. 2016. “Asynchronous Methods for Deep Reinforcement Learning.”

    ## Credit

    Adapted from Ilya Kostrikov's implementation.

    ## Arguments

    * `input_size` (int) - Size of input of the fully connected layers
    * `output_size` (int) - Size of the action space.
    """

    def __init__(self, input_size, output_size):
        super(NatureActor, self).__init__(input_size, output_size)
        atari_init_(self, gain=0.01)


class NatureCritic(nn.Linear):

    """
    <a href="https://github.com/seba-1511/cherry/blob/master/cherry/models/atari.py" class="source-link">[Source]</a>

    ## Description

    The critic head of the A3C architecture.

    ## References

    1. Mnih et al. 2015. “Human-Level Control through Deep Reinforcement Learning.”
    2. Mnih et al. 2016. “Asynchronous Methods for Deep Reinforcement Learning.”

    ## Credit

    Adapted from Ilya Kostrikov's implementation.
    """

    def __init__(self, input_size, output_size=1):
        """
        ## Arguments

        * `input_size` (int) - Size of input of the fully connected layers
        * `output_size` (int, *optional*, default=1) - Size of the value.
        """
        super(NatureCritic, self).__init__(input_size, output_size)
        atari_init_(self, gain=1.0)
