#!/usr/bin/env python3

import random
import gym
import numpy as np

import torch as th
import torch.optim as optim
import torch.nn as nn

import cherry as ch
import cherry.envs as envs
import cherry.distributions as distributions

from cherry.algorithms import a2c
from cherry.models import atari

GAMMA = 0.99
V_WEIGHT = 0.5
ENT_WEIGHT = 0.01
LR = 7e-4
GRAD_NORM = 0.5
A2C_STEPS = 5 * 16


class NatureCNN(nn.Module):

    def __init__(self, env, hidden_size=512):
        super(NatureCNN, self).__init__()
        self.input_size = 4
        self.features = atari.NatureFeatures(self.input_size, hidden_size)
        self.critic = atari.NatureCritic(hidden_size)
        self.actor = atari.NatureActor(hidden_size, env.action_size)
        self.action_dist = distributions.ActionDistribution(env, use_probs=False)

    def forward(self, x):
        x = x.view(-1, self.input_size, 84, 84).mul(1 / 255.0)
        features = self.features(x)
        value = self.critic(features)
        density = self.actor(features)
        mass = self.action_dist(density)
        return mass, value


def update(replay, optimizer, policy, env):
    # Compute advantages
    _, next_state_value = policy(replay[-1].next_state)
    rewards = ch.rewards.discount(GAMMA,
                                  replay.reward(),
                                  replay.done(),
                                  bootstrap=next_state_value)
    rewards = rewards.detach()

    # Compute loss
    entropy = replay.entropy().mean()
    advantages = rewards.detach() - replay.value().detach()
    policy_loss = a2c.policy_loss(replay.log_prob(), advantages)
    value_loss = a2c.state_value_loss(replay.value(), rewards)
    loss = policy_loss + V_WEIGHT * value_loss - ENT_WEIGHT * entropy

    # Take optimization step
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), GRAD_NORM)
    optimizer.step()

    env.log('policy loss', policy_loss.item())
    env.log('value loss', value_loss.item())
    env.log('entropy', entropy.item())


def get_action_value(state, policy):
    mass, value = policy(state)
    action = mass.sample()
    info = {
        'log_prob': mass.log_prob(action),  # Cache log_prob for later
        'value': value,
        'entropy': mass.entropy(),
    }
    return action, info


def main(env='PongNoFrameskip-v4'):
    num_steps = 5000000
    seed = 42

    th.set_num_threads(1)
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env)
    env = envs.Logger(env, interval=1000)
    env = envs.OpenAIAtari(env)
    env = envs.Torch(env)
    env = envs.Runner(env)
    env.seed(seed)

    policy = NatureCNN(env)
    optimizer = optim.RMSprop(policy.parameters(), lr=LR, alpha=0.99, eps=1e-5)
    get_action = lambda state: get_action_value(state, policy)

    for step in range(num_steps // A2C_STEPS + 1):
        # Sample some transitions
        replay = env.run(get_action, steps=A2C_STEPS)

        # Update policy
        update(replay, optimizer, policy, env=env)


if __name__ == '__main__':
    env = 'BreakoutNoFrameskip-v4'
    env = 'PongNoFrameskip-v4'
    main(env)
