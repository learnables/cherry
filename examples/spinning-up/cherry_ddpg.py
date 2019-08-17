#!/usr/bin/env python3

import copy
import random
import numpy as np
import gym
import torch
from torch import optim
from torch import nn

import cherry as ch
from cherry import envs

ACTION_NOISE = 0.1
DISCOUNT = 0.99
HIDDEN_SIZE = 32
LEARNING_RATE = 0.001
MAX_STEPS = 100000
BATCH_SIZE = 128
REPLAY_SIZE = 100000
UPDATE_INTERVAL = 1
UPDATE_START = 10000
POLYAK_FACTOR = 0.995
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def create_target_network(network):
    target_network = copy.deepcopy(network)
    for param in target_network.parameters():
        param.requires_grad = False
    return target_network


class Actor(nn.Module):
    def __init__(self, hidden_size, stochastic=True, layer_norm=False):
        super().__init__()
        layers = [nn.Linear(3, hidden_size),
                  nn.Tanh(),
                  nn.Linear(hidden_size, hidden_size),
                  nn.Tanh(),
                  nn.Linear(hidden_size, 1)]
        if layer_norm:
            layers = (layers[:1]
                      + [nn.LayerNorm(hidden_size)]
                      + layers[1:3]
                      + [nn.LayerNorm(hidden_size)]
                      + layers[3:])
        self.policy = nn.Sequential(*layers)
        if stochastic:
            self.policy_log_std = nn.Parameter(torch.tensor([[0.]]))

    def forward(self, state):
        policy = self.policy(state)
        return policy


class Critic(nn.Module):
    def __init__(self, hidden_size, state_action=False, layer_norm=False):
        super().__init__()
        self.state_action = state_action
        layers = [nn.Linear(3 + (1 if state_action else 0), hidden_size),
                  nn.Tanh(),
                  nn.Linear(hidden_size, hidden_size),
                  nn.Tanh(),
                  nn.Linear(hidden_size, 1)]
        if layer_norm:
            layers = (layers[:1]
                      + [nn.LayerNorm(hidden_size)]
                      + layers[1:3]
                      + [nn.LayerNorm(hidden_size)]
                      + layers[3:])
        self.value = nn.Sequential(*layers)

    def forward(self, state, action=None):
        if self.state_action:
            value = self.value(torch.cat([state, action], dim=1))
        else:
            value = self.value(state)
        return value.squeeze(dim=1)


def get_random_action(state):
    return torch.tensor([[2 * random.random() - 1]])


def main(env='Pendulum-v0'):
    env = gym.make(env)
    env.seed(SEED)
    env = envs.Torch(env)
    env = envs.Logger(env)
    env = envs.Runner(env)

    actor = Actor(HIDDEN_SIZE, stochastic=False, layer_norm=True)
    critic = Critic(HIDDEN_SIZE, state_action=True, layer_norm=True)
    target_actor = create_target_network(actor)
    target_critic = create_target_network(critic)
    actor_optimiser = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critic_optimiser = optim.Adam(critic.parameters(), lr=LEARNING_RATE)
    replay = ch.ExperienceReplay()

    get_action = lambda s: (actor(s) + ACTION_NOISE * torch.randn(1, 1)).clamp(-1, 1)

    for step in range(1, MAX_STEPS + 1):
        with torch.no_grad():
            if step < UPDATE_START:
                replay += env.run(get_random_action, steps=1)
            else:
                replay += env.run(get_action, steps=1)

        replay = replay[-REPLAY_SIZE:]
        if step > UPDATE_START and step % UPDATE_INTERVAL == 0:
            sample = random.sample(replay, BATCH_SIZE)
            batch = ch.ExperienceReplay(sample)

            next_values = target_critic(batch.next_state(),
                                        target_actor(batch.next_state())
                                        ).view(-1, 1)
            values = critic(batch.state(), batch.action()).view(-1, 1)
            value_loss = ch.algorithms.ddpg.state_value_loss(values,
                                                             next_values.detach(),
                                                             batch.reward(),
                                                             batch.done(),
                                                             DISCOUNT)
            critic_optimiser.zero_grad()
            value_loss.backward()
            critic_optimiser.step()

            # Update policy by one step of gradient ascent
            policy_loss = -critic(batch.state(), actor(batch.state())).mean()
            actor_optimiser.zero_grad()
            policy_loss.backward()
            actor_optimiser.step()

            # Update target networks
            ch.models.polyak_average(target_critic,
                                     critic,
                                     POLYAK_FACTOR)
            ch.models.polyak_average(target_actor,
                                     actor,
                                     POLYAK_FACTOR)

if __name__ == '__main__':
    main()
