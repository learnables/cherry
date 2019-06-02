#!/usr/bin/env python3

"""
Simple example of using cherry to solve cartpole.

The code is an adaptation of the PyTorch reinforcement learning example.

TODO: This is not reinforce, this is policy gradient.
"""

import random
import gym
import numpy as np

from itertools import count

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import cherry as ch
import cherry.envs as envs

SEED = 567
GAMMA = 0.99
RENDER = False

random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


def update(replay):
    policy_loss = []

    # Discount and normalize rewards
    rewards = ch.discount(GAMMA, replay.reward(), replay.done())
    rewards = ch.normalize(rewards)

    # Compute loss
    for sars, reward in zip(replay, rewards):
        log_prob = sars.log_prob
        policy_loss.append(-log_prob * reward)

    # Take optimization step
    optimizer.zero_grad()
    policy_loss = th.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = envs.Logger(env, interval=1000)
    env = envs.Torch(env)
    env.seed(SEED)

    policy = PolicyNet()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    running_reward = 10.0
    replay = ch.ExperienceReplay()

    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            mass = Categorical(policy(state))
            action = mass.sample()
            old_state = state
            state, reward, done, _ = env.step(action)
            replay.append(old_state,
                          action,
                          reward,
                          state,
                          done,
                          # Cache log_prob for later
                          log_prob=mass.log_prob(action))
            if RENDER:
                env.render()
            if done:
                break

        #  Compute termination criterion
        running_reward = running_reward * 0.99 + t * 0.01
        if running_reward > env.spec.reward_threshold:
            print('Solved! Running reward is now {} and '
                  'the last episode runs to {} time steps!'.format(running_reward, t))
            break

        # Update policy
        update(replay)
        replay.empty()

