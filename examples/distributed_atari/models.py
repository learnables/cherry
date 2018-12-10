#!/usr/bin/env python3

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

"""
This model is largely inspired from Ilya Kostrikov's implementation:

https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/

TODO: 
    * His implementation uses OpenAI's fancy pre-processing of Atari images. We should include that.
"""


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NatureCNN(nn.Module):

    def __init__(self, num_inputs=3, num_outputs=2, hidden_size=512):
        super(NatureCNN, self).__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu')
        )
        self.features = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0)
        )
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=0.01)
        self.actor_linear = init_(nn.Linear(hidden_size, num_outputs))

        self.train()

    def forward(self, inputs):
        # Inputs should be 1, 4, 84, 84
        inputs = inputs[:, 105-42:105+42, 38:122, :]
        inputs = inputs.permute(0, 3, 1, 2)
        features = self.features(inputs / 255.0)
        value = self.critic_linear(features)
        action_scores = self.actor_linear(features)
        action_mass = Categorical(F.softmax(action_scores, dim=1))
        return action_mass, value
