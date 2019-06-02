#!/usr/bin/env python3

import ppt

import random
import argparse
import gym
import numpy as np

import torch as th
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist

import cherry as ch
import cherry.envs as envs
import cherry.distributions as distributions

from cherry.optim import Distributed
from cherry.algorithms import a2c
from cherry.models import atari

from statistics import mean

"""
This is a demonstration of how to use cherry to train an agent in a distributed
setting.
"""

GAMMA = 0.99
V_WEIGHT = 0.5
ENT_WEIGHT = 0.01
LR = 7e-4
GRAD_NORM = 0.5
A2C_STEPS = 5


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
    rewards = ch.discount(GAMMA,
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

    if dist.get_rank() == 0:
        env.log('policy loss', policy_loss.item())
        env.log('value loss', value_loss.item())
        env.log('entropy', entropy.item())
        ppt.plot(policy_loss.item(), 'A2C - Policy Loss')
        ppt.plot(value_loss.item(), 'A2C - Value Loss')
        ppt.plot(entropy.item(), 'A2C - Entropy')


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

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    dist.init_process_group('gloo',
   			    init_method='file:///home/seba-1511/.dist_init_' + env,
			    rank=args.local_rank,
			    world_size=16)

    rank = dist.get_rank()
    th.set_num_threads(1)
    random.seed(seed + rank)
    th.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    env = gym.make(env)
    if rank == 0:
        env = envs.Logger(env, interval=1000)
    env = envs.OpenAIAtari(env)
    env = envs.Torch(env)
    env = envs.Runner(env)
    env.seed(seed + rank)

    policy = NatureCNN(env)
    optimizer = optim.RMSprop(policy.parameters(), lr=LR, alpha=0.99, eps=1e-5)
    optimizer = Distributed(policy.parameters(), optimizer)
    get_action = lambda state: get_action_value(state, policy)

    for step in range(num_steps // A2C_STEPS + 1):
        # Sample some transitions
        replay = env.run(get_action, steps=A2C_STEPS)

        # Update policy
        update(replay, optimizer, policy, env=env)


if __name__ == '__main__':
    env = 'BreakoutNoFrameskip-v4'
    main(env)
