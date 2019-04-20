#!/usr/bin/env python3

import copy
import random
import gym
import numpy as np

import torch as th
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F

import cherry as ch
import cherry.envs as envs
from cherry.models import atari

"""
This is a demonstration of how to use cherry to train an agent with Deep
Q-Learning.

WARNING: This implementation does not work just yet.
"""

LR = 0.00025
GAMMA = 0.99
EPSILON = 1.0
BSZ = 32
GRAD_NORM = 5.0
UPDATE_FREQ = 1
TARGET_UPDATE_FREQ = 40000
EXPLORATION_STEPS = 50000
EXPLORATION_STEPS = 32
REPLAY_SIZE = 1000000

import ppt
mean = lambda x: sum(x) / len(x)

def epsilon_greedy(q_values, epsilon):
    if random.random() < epsilon:
        actions = th.randint(low=0,
                             high=q_values.size(1),
                             size=(q_values.size(0), 1))
    else:
        actions = q_values.max(1)[1]
    return actions.view(-1, 1)


class DQN(nn.Module):

    def __init__(self, env, hidden_size=512):
        super(DQN, self).__init__()
        self.input_size = 4
        self.features = atari.NatureFeatures(self.input_size, hidden_size)
        self.q = atari.NatureActor(hidden_size, env.action_size)

    def forward(self, x):
        x = x.view(-1, self.input_size, 84, 84).mul(1 / 255.0)
        features = self.features(x)
        q_values = self.q(features)
        return q_values


def update(replay, optimizer, dqn, target_dqn, env):
    batch = replay.sample(BSZ)
    target_q = target_dqn(batch.next_state()).detach().max(dim=1)[0].view(-1, 1)
    target_q = batch.reward() + GAMMA * target_q * (1.0 - batch.done())
    q_preds = dqn(batch.state())
    softnorm = F.softmax(q_preds, dim=1).norm(p=2).item()
    actions = batch.action().view(-1).long()
    range_tensor = th.Tensor(list(range(BSZ))).long()
    q_preds = q_preds[range_tensor, actions]
    loss = (target_q - q_preds).pow(2).mul(0.5).mean()
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(dqn.parameters(), GRAD_NORM)
    optimizer.step()
    env.log('td loss', loss.item())
    env.log('softnorm', softnorm)
    ppt.plot(loss.item(), 'loss')
    ppt.plot(mean(env.all_rewards[-10000:]), 'rewards')
    ppt.plot(mean(env.values['softnorm'][-10000:]), 'softnorm')


def main(num_steps=10000000,
         env_name='PongNoFrameskip-v4',
#         env_name='BreakoutNoFrameskip-v4',
         seed=42):
    th.set_num_threads(1)
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name)
    env = envs.Logger(env, interval=1000)
    env = envs.OpenAIAtari(env)
    env = envs.Torch(env)
    env = envs.Runner(env)
    env.seed(seed)

    dqn = DQN(env)
    target_dqn = copy.deepcopy(dqn)
    optimizer = optim.RMSprop(dqn.parameters(), lr=LR, alpha=0.95,
                              eps=0.01, centered=True)
    replay = ch.ExperienceReplay()
    epsilon = EPSILON
    get_action = lambda state: epsilon_greedy(dqn(state), epsilon)

    for step in range(num_steps // UPDATE_FREQ + 1):
        # Sample some transitions
        ep_replay = env.run(get_action, steps=UPDATE_FREQ)
        replay += ep_replay

        if step * UPDATE_FREQ < 1e6:
            # Update epsilon
            epsilon -= 9.9e-7 * UPDATE_FREQ

        if step * UPDATE_FREQ > EXPLORATION_STEPS:
            # Only keep the last 1M transitions
            replay = replay[-REPLAY_SIZE:]

            # Update Q-function
            update(replay, optimizer, dqn, target_dqn, env=env)

            if step % TARGET_UPDATE_FREQ == 0:
                target_dqn.load_state_dict(dqn.state_dict())


if __name__ == '__main__':
    main()
