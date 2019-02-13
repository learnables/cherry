#!/usr/bin/env python3

import random
import gym
import numpy as np

import torch as th
import torch.optim as optim
import torch.nn as nn

import cherry as ch
import cherry.envs as envs
from cherry.models import atari

"""
This is a demonstration of how to use cherry to train an agent with Deep
Q-Learning.
"""

LR = 0.00025
GAMMA = 0.99
EPSILON = 1.0
BSZ = 32
GRAD_NORM = 5.0
UPDATE_FREQ = 4
TARGET_UPDATE_FREQ = 10000
EXPLORATION_STEPS = 50000
REPLAY_SIZE = 1e6


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


def update(replay, optimizer, dqn, env):
    import pdb; pdb.set_trace()
    return 0.0


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
    optimizer = optim.RMSprop(dqn.parameters(), lr=LR, alpha=0.95,
                              eps=0.01, centered=True)
    replay = ch.ExperienceReplay()
    epsilon = EPSILON
    get_action = lambda state: epsilon_greedy(dqn(state), epsilon)

    for step in range(num_steps // UPDATE_FREQ + 1):
        # Sample some transitions
        num_steps, num_episodes = env.run(get_action, replay, steps=UPDATE_FREQ)

        if step * UPDATE_FREQ < 1e6:
            # Update epsilon
            epsilon -= 9.9e-7 * UPDATE_FREQ

        import pdb; pdb.set_trace()
        if step * UPDATE_FREQ > EXPLORATION_STEPS:
            # Only keep the last 1M transitions
            replay = replay[-REPLAY_SIZE:]
            # Update Q-function
            update(replay, optimizer, dqn, env=env)


if __name__ == '__main__':
    main()

