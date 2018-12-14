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

import cherry as ch
import cherry.envs as envs
import cherry.rewards as ch_rewards

from statistics import mean

from models import NatureCNN
from utils import copy_params, dist_average

from deep_rl.component.envs import make_env

"""
This is a demonstration of how to use cherry to train an agent in a distributed
setting.

Note: It does not aim to replicate any existing implementation.
"""

GAMMA = 0.99
USE_GAE = False
TAU = 1.0
V_WEIGHT = 1.0
ENT_WEIGHT = 0.01
LR = 1e-4
GRAD_NORM = 5.0
NUM_UPDATES = 0


def update(replay, optimizer, policy, shared_params, size, barrier, sync=True, logger=None):
    rewards = replay.list_rewards
    values = [info['value'] for info in replay.list_infos]

    # Discount and normalize rewards
    if USE_GAE:
        _, next_state_value = policy(replay.list_next_states[-1])
        values += [next_state_value]
        rewards, advantages = ch_rewards.gae(GAMMA,
                                             TAU,
                                             rewards,
                                             replay.list_dones,
                                             values,
                                             bootstrap=values[-2])
    else:
        rewards = ch_rewards.discount(GAMMA,
                                      rewards,
                                      replay.list_dones,
                                      bootstrap=values[-1])
        advantages = [r - v for r, v in zip(rewards, values)]

    # Compute losses
    policy_loss = 0.0
    value_loss = 0.0
    entropy_loss = 0.0
    for info, reward, adv in zip(replay.list_infos, rewards, advantages):
        entropy_loss += - info['entropy']
        policy_loss += -info['log_prob'] * adv.item()
        value_loss += 0.5 * (reward.detach() - info['value']).pow(2)

    # Take optimization step
    optimizer.zero_grad()
    loss = policy_loss + V_WEIGHT * value_loss + ENT_WEIGHT * entropy_loss
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), GRAD_NORM)
    optimizer.step()
#    dist_average(policy.parameters(), shared_params, 1.0 / size, barrier, sync)
#    copy_params(shared_params, policy.parameters())

    global NUM_UPDATES
    NUM_UPDATES += 1
    if logger is not None:
        logger.log('policy loss', policy_loss.item())
        logger.log('value loss', value_loss.item())
        logger.log('entropy', entropy_loss.item())
        logger.log('num updates', NUM_UPDATES)


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
        env = envs.Logger(env, interval=1000)
    env = envs.OpenAIAtari(env)
#    env = envs.Atari(env)
#    env = envs.ClipReward(env)
    env = envs.Torch(env)
    env = envs.Runner(env)
    env.seed(seed + rank)

    policy = NatureCNN()
    copy_params(shared_params, policy.parameters())

    optimizer = optim.Adam(policy.parameters(), lr=LR)
    replay = ch.ExperienceReplay()
    get_action = lambda state: get_action_value(state, policy)

    total_num_steps = num_steps
    total_steps = 0
    while total_steps < total_num_steps:
        # Sample some transitions
        num_steps, num_episodes = env.run(get_action, replay, steps=5)

        # Update policy
        update(replay, optimizer, policy, shared_params, size, barrier,
               sync=sync, logger=env)
        replay.empty()
        if rank == 0:
            total_steps += num_steps

    # Save results with randopt
    if rank == 0:
        exp = ro.Experiment('dev', directory='results')
        rewards = env.all_rewards
        dones = env.all_dones
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

    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)
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
            seed,
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
