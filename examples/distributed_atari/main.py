#!/usr/bin/env python3

import os
import random
import gym
import numpy as np
import randopt as ro

import torch as th
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import cherry as ch
import cherry.envs as envs
import cherry.rollouts as rollouts
from cherry.rewards import discount_rewards
from cherry.utils import normalize

from statistics import mean

from models import NatureCNN
from utils import copy_params, dist_average

"""
This is a demonstration of how to use cherry to train an agent in a distributed 
setting.

Note: It does not aim to replicate any existing implementation.
"""

GAMMA = 0.99
V_WEIGHT = 0.5
ENT_WEIGHT = 0.01
LR = 1e-4
GRAD_NORM = 0.5


def update(replay, optimizer, policy, shared_params, size, barrier, sync=True):
    policy_loss = []
    value_loss = []
    entropy_loss = []

    # Discount and normalize rewards
    rewards = discount_rewards(GAMMA, replay.list_rewards, replay.list_dones)
    rewards = normalize(th.tensor(rewards))

    # Compute losses
    for info, reward in zip(replay.list_infos, rewards):
        log_prob = info['log_prob']
        value = info['value']
        entropy = info['entropy']
        entropy_loss.append(-ENT_WEIGHT * entropy)
        policy_loss.append(-log_prob * (reward - value.detach().item()))
        value_loss.append(V_WEIGHT * F.mse_loss(value, reward.detach()))

    # Take optimization step
    optimizer.zero_grad()
    loss = th.stack(policy_loss).sum() + th.stack(value_loss).sum() + th.stack(entropy_loss).sum()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), GRAD_NORM)
    optimizer.step()
    dist_average(policy.parameters(), shared_params, 1.0 / size, barrier, sync)
    copy_params(shared_params, policy.parameters())


def get_action_value(state, policy):
    mass, value = policy(state)
    action = mass.sample()
    info = {
        'log_prob': mass.log_prob(action),  # Cache log_prob for later
        'value': value,
        'entropy': mass.entropy(),
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
    th.set_num_threads(int(os.environ['MKL_NUM_THREADS']))
    random.seed(seed + rank)
    th.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    env = gym.make(env)
    if rank == 0:
        logger = env = envs.Logger(env, interval=5000)
    env = envs.Atari(env)
    env = envs.ClipReward(env)
    env = envs.Torch(env)
    env.seed(seed + rank)

    policy = NatureCNN()
    copy_params(shared_params, policy.parameters())

    optimizer = optim.Adam(policy.parameters(), lr=LR)
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
        if rank == 0:
            total_steps += num_samples

    if rank == 0:
        exp = ro.Experiment('dev', directory='results')
        rewards = logger.all_rewards
        dones = logger.all_dones
        result = mean(rewards[-1000:])
        data = {
            'rewards': rewards,
            'dones': dones,
        }
        exp.add_result(result, data)


@ro.cli
def main(num_workers=2,
         num_steps=10000000,
         env='PongNoFrameskip-v4',
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
        processes = []
        for rank in range(num_workers):
            p = mp.Process(target=run, args=(rank,) + arguments)
            p.start()
            processes.append(p)

        # Wait for main process to finish
        processes[0].join()

        # Kill other workers
        for p in processes[1:]:
            p.terminate()

    else:
        run(0, *arguments)


if __name__ == '__main__':
    ro.parse()
