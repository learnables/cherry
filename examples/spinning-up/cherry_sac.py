#!/usr/bin/env python3

import gym
import copy
import numpy as np
import random
import torch
from torch import optim
from torch import nn
from torch.distributions import Normal, Distribution

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


class TanhNormal(Distribution):
    def __init__(self, loc, scale):
        super().__init__()
        self.normal = Normal(loc, scale)

    def sample(self):
        return torch.tanh(self.normal.sample())

    def rsample(self):
        return torch.tanh(self.normal.rsample())

    def log_prob(self, value):
        inv_value = (torch.log1p(value) - torch.log1p(-value)) / 2.0
        return self.normal.log_prob(inv_value) - torch.log1p(-value.pow(2) + 1e-6)

    @property
    def mean(self):
        return torch.tanh(self.normal.mean)


class SoftActor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.log_std_min, self.log_std_max = -20, 2
        layers = [nn.Linear(3, hidden_size),
                  nn.Tanh(),
                  nn.Linear(hidden_size, hidden_size),
                  nn.Tanh(),
                  nn.Linear(hidden_size, 2)]
        self.policy = nn.Sequential(*layers)

    def forward(self, state):
        policy_mean, policy_log_std = self.policy(state).chunk(2, dim=1)
        policy_log_std = torch.clamp(policy_log_std,
                                     min=self.log_std_min,
                                     max=self.log_std_max)
        policy = TanhNormal(policy_mean, policy_log_std.exp())
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
    replay = ch.ExperienceReplay()

    actor = SoftActor(HIDDEN_SIZE)
    critic_1 = Critic(HIDDEN_SIZE, state_action=True)
    critic_2 = Critic(HIDDEN_SIZE, state_action=True)
    value_critic = Critic(HIDDEN_SIZE)
    target_value_critic = create_target_network(value_critic)
    actor_optimiser = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critics_optimiser = optim.Adam((list(critic_1.parameters())
                                    + list(critic_2.parameters())),
                                   lr=LEARNING_RATE)
    value_critic_optimiser = optim.Adam(value_critic.parameters(),
                                        lr=LEARNING_RATE)
    get_action = lambda state: actor(state).sample()

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

            # Pre-compute some quantities
            states = batch.state()
            rewards = batch.reward()
            old_actions = batch.action()
            dones = batch.done()
            masses = actor(states)
            actions = masses.rsample()
            log_probs = masses.log_prob(actions)
            q_values = torch.min(critic_1(states, actions.detach()),
                                 critic_2(states, actions.detach())
                                 ).view(-1, 1)

            # Compute Q losses
            v_next = target_value_critic(batch.next_state()).view(-1, 1)
            q_old_pred1 = critic_1(states,
                                   old_actions.detach()
                                   ).view(-1, 1)
            q_old_pred2 = critic_2(states,
                                   old_actions.detach()
                                   ).view(-1, 1)
            qloss1 = ch.algorithms.sac.action_value_loss(q_old_pred1,
                                                         v_next.detach(),
                                                         rewards,
                                                         dones,
                                                         DISCOUNT)
            qloss2 = ch.algorithms.sac.action_value_loss(q_old_pred2,
                                                         v_next.detach(),
                                                         rewards,
                                                         dones,
                                                         DISCOUNT)

            # Update Q-functions by one step of gradient descent
            qloss = qloss1 + qloss2
            critics_optimiser.zero_grad()
            qloss.backward()
            critics_optimiser.step()

            # Update V-function by one step of gradient descent
            v_pred = value_critic(batch.state()).view(-1, 1)
            vloss = ch.algorithms.sac.state_value_loss(v_pred,
                                                       log_probs.detach(),
                                                       q_values.detach(),
                                                       alpha=ENTROPY_WEIGHT)
            value_critic_optimiser.zero_grad()
            vloss.backward()
            value_critic_optimiser.step()

            # Update policy by one step of gradient ascent
            q_actions = critic_1(batch.state(), actions).view(-1, 1)
            policy_loss = ch.algorithms.sac.policy_loss(log_probs,
                                                        q_actions,
                                                        alpha=ENTROPY_WEIGHT)
            actor_optimiser.zero_grad()
            policy_loss.backward()
            actor_optimiser.step()

            # Update target value network
            ch.models.polyak_average(target_value_critic,
                                     value_critic,
                                     POLYAK_FACTOR)

if __name__ == '__main__':
    main()
