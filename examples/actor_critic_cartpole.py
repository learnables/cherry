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

import cherry as ch
import cherry.envs as envs
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

    # Logging
    policy_loss = []
    entropies = []
    value_loss = []
    mean = lambda a: (sum(a) / len(a)).mean()

    # Discount and normalize rewards
    rewards = ch.discount(GAMMA, replay.reward(), replay.done())
    rewards = ch.normalize(rewards)

    # Compute losses
    for sars, reward in zip(replay, rewards):
        log_prob = sars.log_prob
        value = sars.value.view(-1).detach()
        policy_loss.append(-log_prob * (reward - value))
        value_loss.append(F.mse_loss(value, reward.detach()))

    # Take optimization step
    optimizer.zero_grad()
    loss = th.stack(policy_loss).sum() + V_WEIGHT * th.stack(value_loss).sum()
    loss.backward()
    optimizer.step()


    # Log metrics
    env.log('policy loss', mean(policy_loss).item())
    #env.log('policy entropy', mean(entropies).item())
    env.log('value loss', mean(value_loss).item())


def get_action_value(state, policy):
    mass, value = policy(state)
    action = mass.sample()
    info = {
        'log_prob': mass.log_prob(action),  # Cache log_prob for later
        'value': value
    }
    return action, info


if __name__ == '__main__':
    env = gym.vector.make('CartPole-v0', num_envs=4)
    env = envs.Logger(env)
    env = envs.Torch(env)
    env = envs.Runner(env)
    env.seed(SEED)

    #if RECORD:
    #    record_env = envs.Monitor(env, './videos/')

    policy = ActorCriticNet(env)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    running_reward = 10.0
    get_action = lambda state: get_action_value(state, policy)

    for episode in range(500):
        # We use the Runner collector, but could've written our own
        replay = env.run(get_action, steps=200)

        # Update policy
        update(replay, optimizer)
