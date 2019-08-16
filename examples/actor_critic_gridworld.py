#!/usr/bin/env python3

"""
Simple example of using cherry to solve cartpole with an actor-critic.

The code is an adaptation of the PyTorch reinforcement learning example.
"""

import random
import gym
import gym_minigrid
import numpy as np

from itertools import count

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cherry as ch
from cherry import envs
from cherry import td
from cherry import distributions

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
        self.affine1 = nn.Linear(env.state_size['image'], 128)
        self.action_head = nn.Linear(128, env.action_size)
        self.value_head = nn.Linear(128, 1)
        self.distribution = distributions.ActionDistribution(env)

    def forward(self, x):
        x = x.view(-1)
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        action_mass = self.distribution(F.log_softmax(action_scores, dim=0))
        value = self.value_head(x)
        return action_mass, value


def update(replay, optimizer, policy):
    policy_loss = []
    value_loss = []

    # Discount and normalize rewards
    rewards = td.discount(GAMMA, replay.reward(), replay.done())
    rewards = ch.normalize(rewards)

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
    env = gym.make('MiniGrid-Empty-6x6-v0')
#    env = gym.make('MiniGrid-LavaCrossingS9N1-v0')
    env = envs.StateLambda(env, lambda x: x['image'])
    env = envs.VisdomLogger(env, interval=1000)
    env = envs.Torch(env)
    env = envs.Runner(env)
    env.seed(SEED)

    policy = ActorCriticNet(env)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    running_reward = 10.0
    get_action = lambda state: get_action_value(state, policy)

    for episode in count(1):
        # We use the Runner collector, but could've written our own
        replay = env.run(get_action, episodes=1, render=RENDER)

        # Update policy
        update(replay, optimizer, policy)
        if episode > 6400:
            RENDER = True

        # Compute termination criterion
        #running_reward = running_reward * 0.99 + len(replay) * 0.01
        #if running_reward > env.spec.reward_threshold:
        #    print('Solved! Running reward now {} and '
        #          'the last episode runs to {} time steps!'.format(running_reward,
        #                                                           len(replay)))
        #    break
