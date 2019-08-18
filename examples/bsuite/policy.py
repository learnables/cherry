#!/usr/bin/env python3

import cherry as ch
from torch import nn


class Policy(nn.Module):

    def __init__(self, env):
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(env.state_size, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, env.action_size)
        self.dist = ch.distributions.ActionDistribution(env)

    def density(self, state):
        x = self.layer1(state)
        x = self.relu(x)
        x = self.layer2(x)
        return self.dist(x)

    def log_prob(self, state, action):
        density = self.density(state)
        return density.log_prob(action)

    def forward(self, state):
        density = self.density(state)
        return density.sample().detach()
