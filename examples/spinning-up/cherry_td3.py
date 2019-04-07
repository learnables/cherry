
import copy
import gym
from collections import deque
import random
import numpy as np
import torch
from torch import optim
from torch import nn

import cherry as ch
from cherry import envs

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
MAX_STEPS = 100000
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

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def create_target_network(network):
    target_network = copy.deepcopy(network)
    for param in target_network.parameters():
        param.requires_grad = False
    return target_network


def update_target_network(network, target_network, polyak_factor):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data = polyak_factor * target_param.data + (1 - polyak_factor) * param.data


class Actor(nn.Module):
    def __init__(self, hidden_size, stochastic=True, layer_norm=False):
        super().__init__()
        layers = [nn.Linear(3, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)]
        if layer_norm:
            layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [nn.LayerNorm(hidden_size)] + layers[3:]  # Insert layer normalisation between fully-connected layers and nonlinearities
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
            layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [nn.LayerNorm(hidden_size)] + layers[3:]  # Insert layer normalisation between fully-connected layers and nonlinearities
        self.value = nn.Sequential(*layers)

    def forward(self, state, action=None):
        if self.state_action:
            value = self.value(torch.cat([state, action], dim=1))
        else:
            value = self.value(state)
        return value.squeeze(dim=1)


def get_random_action(state):
    return torch.tensor([[2 * random.random() - 1]])


def get_action(state):
    action = actor(state) + ACTION_NOISE * torch.randn(1, 1)
    return torch.clamp(action, min=-1, max=1)


env = gym.make('Pendulum-v0')
env.seed(SEED)
env = envs.Torch(env)
env = envs.Runner(env)

actor = Actor(HIDDEN_SIZE, stochastic=False, layer_norm=True)
critic_1 = Critic(HIDDEN_SIZE, state_action=True, layer_norm=True)
critic_2 = Critic(HIDDEN_SIZE, state_action=True, layer_norm=True)
target_actor = create_target_network(actor)
target_critic_1 = create_target_network(critic_1)
target_critic_2 = create_target_network(critic_2)
actor_optimiser = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critics_optimiser = optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()), lr=LEARNING_RATE)
replay = ch.ExperienceReplay()


for step in range(1, MAX_STEPS + 1):
    with torch.no_grad():
        if step < UPDATE_START:
            replay += env.run(get_random_action, steps=1)
        else:
            replay += env.run(get_action, steps=1)

    if step > UPDATE_START and step % UPDATE_INTERVAL == 0:
        sample = random.sample(replay, BATCH_SIZE)
        batch = ch.ExperienceReplay()
        for sars in sample:
            batch.append(**sars)

        next_values = target_critic(batch.next_states, target_actor(batch.next_states)).view(-1, 1)
        values = critic(batch.states, batch.actions).view(-1, 1)
        value_loss = ch.algorithms.ddpg.state_value_loss(values,
                                                         next_values,
                                                         batch.rewards,
                                                         batch.dones,
                                                         DISCOUNT)
        print(step)
        print('vloss', value_loss.item())
        critic_optimiser.zero_grad()
        value_loss.backward()
        critic_optimiser.step()

        # Update policy by one step of gradient ascent
        policy_loss = -critic(batch.states, actor(batch.states)).mean()
        print('ploss', policy_loss.item())
        actor_optimiser.zero_grad()
        policy_loss.backward()
        actor_optimiser.step()

        # Update target networks
        ch.models.polyak_average(target_critic,
                                 critic,
                                 1.0 - POLYAK_FACTOR)
        ch.models.polyak_average(target_actor,
                                 actor,
                                 1.0 - POLYAK_FACTOR)
        print('')
