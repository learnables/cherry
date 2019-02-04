#!/usr/bin/env python3

import random
import gym
import numpy as np

import torch as th
import torch.optim as optim
import torch.nn as nn

import cherry as ch
import cherry.envs as envs
import cherry.mpi as mpi
import cherry.policies as policies
from cherry.models import atari

"""
This is a demonstration of how to use cherry to train an agent in a distributed
setting.
"""

GAMMA = 0.99
USE_GAE = True
TAU = 0.95
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
        self.action_dist = policies.ActionDistribution(env, use_probs=False)

    def forward(self, x):
        x = x.view(-1, self.input_size, 84, 84).mul(1 / 255.0)
        features = self.features(x)
        value = self.critic(features)
        density = self.actor(features)
        mass = self.action_dist(density)
        return mass, value


def update(replay, optimizer, policy, env):
    # Compute advantages
    _, next_state_value = policy(replay.next_states[-1])
    if USE_GAE:
        advantages = ch.rewards.gae(GAMMA,
                                    TAU,
                                    replay.rewards,
                                    replay.dones,
                                    replay.values,
                                    next_state_value)
    else:
        rewards = ch.rewards.discount(GAMMA,
                                      replay.rewards,
                                      replay.dones,
                                      bootstrap=next_state_value)
        advantages = [r.detach() - v for r, v in zip(rewards, replay.values)]

    # Compute loss
    entropy = replay.entropys.mean()
    advantages = th.cat(advantages, dim=0).view(-1, 1)
    policy_loss = - th.mean(replay.log_probs * advantages.detach())
    value_loss = advantages.pow(2).mean()
    loss = policy_loss + V_WEIGHT * value_loss - ENT_WEIGHT * entropy

    # Take optimization step
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), GRAD_NORM)
    optimizer.step()

    if mpi.rank == 0:
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


def main(num_steps=10000000,
         env='PongNoFrameskip-v4',
         seed=1234):
    th.set_num_threads(1)
    random.seed(seed + mpi.rank)
    th.manual_seed(seed + mpi.rank)
    np.random.seed(seed + mpi.rank)

    env = gym.make(env)
    if mpi.rank == 0:
        env = envs.Logger(env, interval=1000)
    env = envs.OpenAIAtari(env)
    env = envs.Torch(env)
    env = envs.Runner(env)
    env.seed(seed + mpi.rank)

    policy = NatureCNN(env)
    optimizer = optim.RMSprop(policy.parameters(), lr=LR, alpha=0.99, eps=1e-5)
    optimizer = mpi.Distributed(policy.parameters(), optimizer)
    replay = ch.ExperienceReplay()
    get_action = lambda state: get_action_value(state, policy)

    for step in range(num_steps // A2C_STEPS + 1):
        # Sample some transitions
        num_steps, num_episodes = env.run(get_action, replay, steps=A2C_STEPS)

        # Update policy
        update(replay, optimizer, policy, env=env)
        replay.empty()

    if mpi.rank == 0:
        # Kill all MPI processes
        mpi.terminate_mpi()
        mpi.comm.Abort()


if __name__ == '__main__':
    main()

