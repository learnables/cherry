#!/usr/bin/env python3

import random
import gym
import numpy as np
import randopt as ro

import torch as th
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F

import cherry as ch
import cherry.envs as envs
import cherry.rollouts as rollouts
from cherry.rewards import discount_rewards
from cherry.utils import normalize

from models import NatureCNN
from utils import copy_params, dist_average

"""
This is a demonstration of how to use cherry to train an agent in a distributed 
setting.

"""

GAMMA = 0.99
V_WEIGHT = 0.1


def update(replay, optimizer, policy, shared_params, size, barrier, sync=True):
    policy_loss = []
    value_loss = []

    # Discount and normalize rewards
    rewards = discount_rewards(GAMMA, replay.list_rewards, replay.list_dones)
    rewards = normalize(th.tensor(rewards))

    # Compute losses
    for info, reward in zip(replay.list_infos, rewards):
        log_prob = info['log_prob']
        value = info['value']
        policy_loss.append(-log_prob * (reward - value.item()))
        value_loss.append(F.mse_loss(value, reward.detach()))

    # Take optimization step
    optimizer.zero_grad()
    loss = th.stack(policy_loss).sum() + V_WEIGHT * th.stack(value_loss).sum()
    loss.backward()
    optimizer.step()
    dist_average(policy.parameters(), shared_params, 1.0 / size, barrier, sync)
    copy_params(shared_params, policy.parameters())


def get_action_value(state, policy):
    mass, value = policy(state)
    action = mass.sample()
    info = {
        'log_prob': mass.log_prob(action),  # Cache log_prob for later
        'value': value
    }
    return action, info


def run(rank,
        size,
        num_steps,
        env,
        barrier,
        shared_params,
        sync=True,
        seed=1234):
    th.set_num_threads(1)
    random.seed(seed + rank)
    th.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    env = gym.make(env)
    env = envs.Atari(env)
    env = envs.ClipReward(env)
    if rank == 0:
        env = envs.Logger(env, interval=5000)
    env = envs.Torch(env)
    env.seed(seed + rank)

    policy = NatureCNN()
    copy_params(shared_params, policy.parameters())

    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    replay = ch.ExperienceReplay()
    get_action = lambda state: get_action_value(state, policy)

    total_steps = 0
    while total_steps < num_steps:
        # We use the rollout collector, but could've written our own
        num_samples, num_episodes = rollouts.collect(env,
                                                     get_action,
                                                     replay,
                                                     num_episodes=1)
        # Update policy
        update(replay,
               optimizer,
               policy,
               shared_params,
               size,
               barrier,
               sync=sync)
        replay.empty()
        total_steps += num_samples


@ro.cli
def main(num_workers=2,
         num_steps=10000000,
         env='PongNoFrameskip-v0',
         sync=True,
         seed=1234):

    manager = mp.Manager()
    shared_policy = NatureCNN()
    shared_params = [p.data.share_memory_()
                     for p in shared_policy.parameters()]

    arguments = (
            num_workers,
            num_steps,
            env,
            manager.Barrier(num_workers),
            shared_params,
            sync,
            seed
            )
    if num_workers > 1:
        mp.spawn(run, arguments, num_workers, join=True)
    else:
        run(0, *arguments)


if __name__ == '__main__':
    ro.parse()
