#!/usr/bin/env python3

import torch as th
import torch.nn as nn
from torch.distributions import Categorical

"""
This model is largely inspired from Ilya Kostrikov's implementation:

https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/
"""


def init(module, weight_init, bias_init, gain=1):
#    weight_init(module.weight.data, gain=gain)
    weight_init(module.weight.data)
    bias_init(module.bias.data)
    return module


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NatureCNN(nn.Module):

    def __init__(self, num_inputs=4, num_outputs=6, hidden_size=512):
        super(NatureCNN, self).__init__()
        self.num_inputs = num_inputs
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
        self.critic_linear.weight.data.mul_(1e-3)

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=0.01)
        self.actor_linear = init_(nn.Linear(hidden_size, num_outputs))
        self.actor_linear.weight.data.mul_(1e-3)

        self.train()

        weights = th.load('./model.pth')
        self.features[0].weight.data.copy_(weights['base.main.0.weight'])
        self.features[0].bias.data.copy_(weights['base.main.0.bias'])
        self.features[2].weight.data.copy_(weights['base.main.2.weight'])
        self.features[2].bias.data.copy_(weights['base.main.2.bias'])
        self.features[4].weight.data.copy_(weights['base.main.4.weight'])
        self.features[4].bias.data.copy_(weights['base.main.4.bias'])
        self.features[7].weight.data.copy_(weights['base.main.7.weight'])
        self.features[7].bias.data.copy_(weights['base.main.7.bias'])
        self.critic_linear.weight.data.copy_(weights['base.critic_linear.weight'])
        self.critic_linear.bias.data.copy_(weights['base.critic_linear.bias'])
        self.actor_linear.weight.data.copy_(weights['dist.linear.weight'])
        self.actor_linear.bias.data.copy_(weights['dist.linear.bias'])

    def forward(self, inputs):
        # Inputs should be 1, 4, 84, 84 for a single state
        inputs = inputs.view(-1, self.num_inputs, 84, 84)
        features = self.features(inputs / 255.0)
        value = self.critic_linear(features)
        action_scores = self.actor_linear(features)
        action_mass = Categorical(logits=action_scores)
        return action_mass, value
