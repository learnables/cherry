#!/usr/bin/env python3

"""
Simple example of using cherry to solve cartpole with an actor-critic.

The code is an adaptation of the PyTorch reinforcement learning example.
"""

import random
import gym
import numpy as np

from itertools import count

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cherry.envs as envs
from cherry.td import discount
from cherry import normalize
import cherry.distributions as distributions

SEED = 567
GAMMA = 0.99
RENDER = False
V_WEIGHT = 0.5

random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)


class ActorCriticNet(nn.Module):
    def __init__(self, env):
        super(ActorCriticNet, self).__init__()
        self.affine1 = nn.Linear(env.state_size, 128)
        self.action_head = nn.Linear(128, env.action_size)
        self.value_head = nn.Linear(128, 1)
        self.distribution = distributions.ActionDistribution(env,
                                                             use_probs=True)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        action_mass = self.distribution(F.softmax(action_scores, dim=1))
        value = self.value_head(x)
        return action_mass, value


def update(replay, optimizer):
    policy_loss = []
    value_loss = []

    # Discount and normalize rewards
    rewards = discount(GAMMA, replay.reward(), replay.done())
    rewards = normalize(rewards)

    # Compute losses
    for sars, reward in zip(replay, rewards):
        log_prob = sars.log_prob
        value = sars.value
        policy_loss.append(-log_prob * (reward - value.item()))
        value_loss.append(F.mse_loss(value, reward.detach()))

    # Take optimization step
    optimizer.zero_grad()
    loss = th.stack(policy_loss).sum() + V_WEIGHT * th.stack(value_loss).sum()
    loss.backward()
    optimizer.step()


def get_action_value(state, policy):
    mass, value = policy(state)
    action = mass.sample()
    info = {
        'log_prob': mass.log_prob(action),  # Cache log_prob for later
        'value': value
    }
    return action, info


if __name__ == '__main__':
    env = gym.vector.make('CartPole-v0', num_envs=1)
    env = envs.Logger(env, interval=1000)
    env = envs.Torch(env)
    env = envs.Runner(env)
    env.seed(SEED)

    policy = ActorCriticNet(env)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    running_reward = 10.0
    get_action = lambda state: get_action_value(state, policy)

    for episode in count(1):
        # We use the Runner collector, but could've written our own
        replay = env.run(get_action, episodes=1)

        # Update policy
        update(replay, optimizer)

        # Compute termination criterion
        running_reward = running_reward * 0.99 + len(replay) * 0.01
        if episode % 10 == 0:
            # Should start with 10.41, 12.21, 14.60, then 100:71.30, 200:135.74
            print(episode, running_reward)
        if running_reward > 190.0:
            print('Solved! Running reward now {} and '
                  'the last episode runs to {} time steps!'.format(running_reward,
                                                                   len(replay)))
            break
