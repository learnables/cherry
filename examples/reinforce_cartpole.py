#!/usr/bin/env python3

"""
Simple example of using cherry to solve cartpole.

The code is an adaptation of the PyTorch reinforcement learning example.
"""

import gym

from itertools import count

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import cherry as cr
from cherry.utils import normalize

SEED = 1234
GAMMA = 0.99
RENDER = False


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


def select_action(state):
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item()


def finish_episode(replay):
    R = 0
    policy_loss = []
    rewards = []
    # Compute discounted rewards
    for r in replay.list_rewards[::-1]:
        R = r + GAMMA * R
        rewards.insert(0, R)

    # Normalize rewards
    rewards = th.tensor(rewards)
    rewards = normalize(rewards)

    for state, action, reward in zip(replay.list_states,
                                     replay.list_actions,
                                     rewards):
        # Compute log_prob of action
        probs = policy(state)
        m = Categorical(probs)
        log_prob = m.log_prob(action)

        # Compute loss
        policy_loss.append(-log_prob * reward)

    optimizer.zero_grad()
    policy_loss = th.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(SEED)
    th.manual_seed(SEED)

    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    running_reward = 10.0
    replay = cr.ExperienceReplay()

    for i_episode in count(1):
        state = env.reset()
        state = th.from_numpy(state).float().unsqueeze(0)
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            old_state = state
            state, reward, done, _ = env.step(action)
            state = th.from_numpy(state).float().unsqueeze(0)
            replay.add(old_state, action, reward, state, done)
            if RENDER:
                env.render()
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode(replay)
        replay.empty()
        if i_episode % 10 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                  i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print('Solved! Running reward is now {} and '
                  'the last episode runs to {} time steps!'.format(running_reward, t))
            break
