#!/usr/bin/env python3

import unittest

import copy
import torch
import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
import gym
from collections import deque
from torch import optim
from torch import nn
from torch.distributions import Normal
import cherry as ch
from cherry import envs

TOL = 1e-8

ACTION_DISCRETISATION = 5
ACTION_NOISE = 0.1
BACKTRACK_COEFF = 0.8
BACKTRACK_ITERS = 10
CONJUGATE_GRADIENT_ITERS = 10
DAMPING_COEFF = 0.1
DISCOUNT = 0.99
EPSILON = 0.05
ENTROPY_WEIGHT = 0.2
HIDDEN_SIZE = 32
KL_LIMIT = 0.05
LEARNING_RATE = 0.001
MAX_STEPS = 11000
ON_POLICY_BATCH_SIZE = 2048
BATCH_SIZE = 128
POLICY_DELAY = 2
POLYAK_FACTOR = 0.995
PPO_CLIP_RATIO = 0.2
PPO_EPOCHS = 20
REPLAY_SIZE = 100000
TARGET_ACTION_NOISE = 0.2
TARGET_ACTION_NOISE_CLIP = 0.5
TARGET_UPDATE_INTERVAL = 2500
TRACE_DECAY = 0.97
UPDATE_INTERVAL = 1
UPDATE_START = 10000
TEST_INTERVAL = 1000
SEED = 42


class Env():
    def __init__(self):
        self._env = gym.make('Pendulum-v1')
        self._env.seed(SEED)

    def reset(self):
        state = self._env.reset()
        return torch.tensor(state, dtype=torch.float64).unsqueeze(dim=0)

    def step(self, action):
        state, reward, done, _ = self._env.step(action[0].detach().numpy())
        state = torch.tensor(state, dtype=torch.float64).unsqueeze(dim=0)
        return state, reward, done


def create_target_network(network):
    target_network = copy.deepcopy(network)
    for param in target_network.parameters():
        param.requires_grad = False
    return target_network


def update_target_network(network, target_network, polyak_factor):
    for param, target_param in zip(network.parameters(),
                                   target_network.parameters()):
        target_param.data = polyak_factor * target_param.data + (1 - polyak_factor) * param.data


class Actor(nn.Module):
    def __init__(self, hidden_size, stochastic=True, layer_norm=False):
        super().__init__()
        layers = [nn.Linear(3, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)]
        if layer_norm:
            layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [nn.LayerNorm(hidden_size)] + layers[3:]
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
        layers = [nn.Linear(3 + (1 if state_action else 0), hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)]
        if layer_norm:
            layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [nn.LayerNorm(hidden_size)] + layers[3:]
        self.value = nn.Sequential(*layers)

    def forward(self, state, action=None):
        if self.state_action:
            value = self.value(torch.cat([state, action], dim=1))
        else:
            value = self.value(state)
        return value.squeeze(dim=1)


def train_spinup():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    result = {
        'rewards': [],
        'plosses': [],
        'vlosses': [],
        'pweights': [],
        'vweights': [],
        'target_vweights': [],
        'target_pweights': [],
    }

    env = Env()
    actor = Actor(HIDDEN_SIZE, stochastic=False, layer_norm=True)
    critic = Critic(HIDDEN_SIZE, state_action=True, layer_norm=True)
    target_actor = create_target_network(actor)
    target_critic = create_target_network(critic)
    actor_optimiser = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critic_optimiser = optim.Adam(critic.parameters(), lr=LEARNING_RATE)
    D = deque(maxlen=REPLAY_SIZE)

    state, done = env.reset(), False
    for step in range(1, MAX_STEPS + 1):
        with torch.no_grad():
            if step < UPDATE_START:
                action = torch.tensor([[2 * random.random() - 1]])
            else:
                action = torch.clamp(actor(state) + ACTION_NOISE * torch.randn(1, 1),
                                     min=-1,
                                     max=1)
            next_state, reward, done = env.step(action)
            D.append({'state': state,
                      'action': action,
                      'reward': torch.tensor([reward]),
                      'next_state': next_state,
#                      'done': torch.tensor([done], dtype=torch.float32)})
                      'done': torch.tensor([done], dtype=torch.float64)})
            state = next_state
            if done:
                state = env.reset()
            result['rewards'].append(reward)

        if step > UPDATE_START and step % UPDATE_INTERVAL == 0:
            batch = random.sample(D, BATCH_SIZE)
            batch = {k: torch.cat([d[k] for d in batch], dim=0) for k in batch[0].keys()}
            y = batch['reward'] + DISCOUNT * (1 - batch['done']) * target_critic(batch['next_state'], target_actor(batch['next_state']))

            # Update Q-function by one step of gradient descent
            value_loss = (critic(batch['state'], batch['action']) - y).pow(2).mean()
            critic_optimiser.zero_grad()
            value_loss.backward()
            critic_optimiser.step()
            result['vlosses'].append(value_loss.item())

            # Update policy by one step of gradient ascent
            policy_loss = -critic(batch['state'], actor(batch['state'])).mean()
            actor_optimiser.zero_grad()
            policy_loss.backward()
            actor_optimiser.step()
            result['plosses'].append(policy_loss.item())

            # Update target networks
            update_target_network(critic, target_critic, POLYAK_FACTOR)
            update_target_network(actor, target_actor, POLYAK_FACTOR)

    result['pweights'] = list(actor.parameters())
    result['target_pweights'] = list(target_actor.parameters())
    result['vweights'] = list(critic.parameters())
    result['target_vweights'] = list(target_critic.parameters())
    return result


def train_cherry():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    result = {
        'rewards': [],
        'plosses': [],
        'vlosses': [],
        'pweights': [],
        'vweights': [],
        'target_vweights': [],
        'target_pweights': [],
    }

    env = gym.make('Pendulum-v1')
    env.seed(SEED)
    env = envs.Torch(env)
    env = envs.Runner(env)

    actor = Actor(HIDDEN_SIZE, stochastic=False, layer_norm=True)
    critic = Critic(HIDDEN_SIZE, state_action=True, layer_norm=True)
    target_actor = create_target_network(actor)
    target_critic = create_target_network(critic)
    actor_optimiser = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critic_optimiser = optim.Adam(critic.parameters(), lr=LEARNING_RATE)
    replay = ch.ExperienceReplay()

    def get_random_action(state):
        return torch.tensor([[2 * random.random() - 1]])

    def get_action(state):
        action = actor(state) + ACTION_NOISE * torch.randn(1, 1)
        return torch.clamp(action, min=-1, max=1)

    for step in range(1, MAX_STEPS + 1):
        with torch.no_grad():
            if step < UPDATE_START:
                replay += env.run(get_random_action, steps=1)
            else:
                replay += env.run(get_action, steps=1)

        result['rewards'].append(replay.reward()[-1].item())
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
            result['vlosses'].append(value_loss.item())

            # Update policy by one step of gradient ascent
            policy_loss = -critic(batch.state(), actor(batch.state())).mean()
            actor_optimiser.zero_grad()
            policy_loss.backward()
            actor_optimiser.step()
            result['plosses'].append(policy_loss.item())

            # Update target networks
            ch.models.polyak_average(target_critic,
                                     critic,
                                     POLYAK_FACTOR)
            ch.models.polyak_average(target_actor,
                                     actor,
                                     POLYAK_FACTOR)

    result['pweights'] = list(actor.parameters())
    result['target_pweights'] = list(target_actor.parameters())
    result['vweights'] = list(critic.parameters())
    result['target_vweights'] = list(target_critic.parameters())
    return result


def close(a, b):
    return (a-b).norm(p=2) <= TOL


class TestSpinningUpDDPG(unittest.TestCase):

    def setUp(self):
        torch.set_default_tensor_type(torch.DoubleTensor)
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        torch.set_default_tensor_type(torch.FloatTensor)
        torch.set_default_dtype(torch.float32)

    def test_ddpg(self):
        cherry = train_cherry()
        spinup = train_spinup()

        for key in cherry.keys():
            if key == 'rewards':
                continue
            self.assertTrue(len(cherry[key]) > 0)
            self.assertTrue(len(spinup[key]) > 0)
            for cv, sv in zip(cherry[key], spinup[key]):
                if isinstance(cv, torch.Tensor):
                    self.assertTrue(close(cv, sv))
                else:
                    self.assertTrue(abs(cv - sv) <= TOL)


if __name__ == "__main__":
    unittest.main()
