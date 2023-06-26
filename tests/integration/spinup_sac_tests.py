#!/usr/bin/env python3

import unittest

import numpy as np
import torch
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
import gym
import copy
from collections import deque
from torch import optim
from torch import nn
from torch.distributions import Normal, Distribution
import cherry as ch
from cherry import envs

TOL = 1e-6

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


def create_target_network(network):
    target_network = copy.deepcopy(network)
    for param in target_network.parameters():
        param.requires_grad = False
    return target_network


def update_target_network(network, target_network, polyak_factor):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data = polyak_factor * target_param.data + (1 - polyak_factor) * param.data


class TanhNormal(Distribution):
    def __init__(self, loc, scale):
        super().__init__()
        self.normal = Normal(loc, scale)

    def sample(self):
        return torch.tanh(self.normal.sample())

    def rsample(self):
        return torch.tanh(self.normal.rsample())

    # Calculates log probability of value using the change-of-variables technique (uses log1p = log(1 + x) for extra numerical stability)
    def log_prob(self, value):
        inv_value = (torch.log1p(value) - torch.log1p(-value)) / 2  # artanh(y)
        return self.normal.log_prob(inv_value) - torch.log1p(-value.pow(2) + 1e-6)  # log p(f^-1(y)) + log |det(J(f^-1(y)))|

    @property
    def mean(self):
        return torch.tanh(self.normal.mean)


class SoftActor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.log_std_min, self.log_std_max = -20, 2
        layers = [nn.Linear(3, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 2)]
        self.policy = nn.Sequential(*layers)

    def forward(self, state):
        policy_mean, policy_log_std = self.policy(state).chunk(2, dim=1)
        policy_log_std = torch.clamp(policy_log_std, min=self.log_std_min, max=self.log_std_max)
        policy = TanhNormal(policy_mean, policy_log_std.exp())
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


def train_spinup():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    result = {
        'rewards': [],
        'plosses': [],
        'vlosses': [],
        'qlosses': [],
        'pweights': [],
        'vweights': [],
        'vweights_target': [],
        'qweights1': [],
        'qweights2': [],
    }

    env = Env()

    actor = SoftActor(HIDDEN_SIZE)
    critic_1 = Critic(HIDDEN_SIZE, state_action=True)
    critic_2 = Critic(HIDDEN_SIZE, state_action=True)
    value_critic = Critic(HIDDEN_SIZE)
    target_value_critic = create_target_network(value_critic)
    actor_optimiser = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critics_optimiser = optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()), lr=LEARNING_RATE)
    value_critic_optimiser = optim.Adam(value_critic.parameters(), lr=LEARNING_RATE)
    D = deque(maxlen=REPLAY_SIZE)

    state, done = env.reset(), False
    for step in range(1, MAX_STEPS + 1):
        with torch.no_grad():
            if step < UPDATE_START:
                action = torch.tensor([[2 * random.random() - 1]])
            else:
                action = actor(state).sample()
            next_state, reward, done = env.step(action)
            D.append({'state': state,
                      'action': action,
                      'reward': torch.tensor([reward]),
                      'next_state': next_state,
                      'done': torch.tensor([done], dtype=torch.float64)})
            state = next_state
            if done:
                state = env.reset()
#            result['rewards'].append(reward)
            result['rewards'].append(D[-1]['reward'].item())

        if step > UPDATE_START and step % UPDATE_INTERVAL == 0:
            batch = random.sample(D, BATCH_SIZE)
            batch = {k: torch.cat([d[k] for d in batch], dim=0) for k in batch[0].keys()}
            y_q = batch['reward'] + DISCOUNT * (1 - batch['done']) * target_value_critic(batch['next_state'])
            policy = actor(batch['state'])
            action = policy.rsample()
            weighted_sample_entropy = ENTROPY_WEIGHT * policy.log_prob(action).sum(dim=1)
            y_v = torch.min(critic_1(batch['state'], action), critic_2(batch['state'], action)).detach() - weighted_sample_entropy.detach()

            # Update Q-functions by one step of gradient descent
            value_loss = (critic_1(batch['state'], batch['action']) - y_q).pow(2).mean() + (critic_2(batch['state'], batch['action']) - y_q).pow(2).mean()
            critics_optimiser.zero_grad()
            value_loss.backward()
            critics_optimiser.step()
            result['qlosses'].append(value_loss.item())

            # Update V-function by one step of gradient descent
            value_loss = (value_critic(batch['state']) - y_v).pow(2).mean()
            value_critic_optimiser.zero_grad()
            value_loss.backward()
            value_critic_optimiser.step()
            result['vlosses'].append(value_loss.item())

            # Update policy by one step of gradient ascent
            policy_loss = (weighted_sample_entropy - critic_1(batch['state'], action)).mean()
            actor_optimiser.zero_grad()
            policy_loss.backward()
            actor_optimiser.step()
            result['plosses'].append(policy_loss.item())

            # Update target value network
            update_target_network(value_critic, target_value_critic, POLYAK_FACTOR)

    result['pweights'] = list(actor.parameters())
    result['vweights'] = list(value_critic.parameters())
    result['vweights_target'] = list(target_value_critic.parameters())
    result['qweights1'] = list(critic_1.parameters())
    result['qweights2'] = list(critic_2.parameters())
    return result


def train_cherry():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    result = {
        'rewards': [],
        'plosses': [],
        'vlosses': [],
        'qlosses': [],
        'pweights': [],
        'vweights': [],
        'vweights_target': [],
        'qweights1': [],
        'qweights2': [],
    }

    env = gym.make('Pendulum-v1')
    env.seed(SEED)
    env = envs.Torch(env)
    env = envs.Runner(env)
    replay = ch.ExperienceReplay()

    actor = SoftActor(HIDDEN_SIZE)
    critic_1 = Critic(HIDDEN_SIZE, state_action=True)
    critic_2 = Critic(HIDDEN_SIZE, state_action=True)
    value_critic = Critic(HIDDEN_SIZE)
    target_value_critic = create_target_network(value_critic)
    actor_optimiser = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critics_optimiser = optim.Adam(list(critic_1.parameters()) + list(critic_2.parameters()), lr=LEARNING_RATE)
    value_critic_optimiser = optim.Adam(value_critic.parameters(), lr=LEARNING_RATE)

    def get_random_action(state):
        return torch.tensor([[2 * random.random() - 1]])

    def get_action(state):
        return actor(state).sample()

    for step in range(1, MAX_STEPS + 1):
        with torch.no_grad():
            if step < UPDATE_START:
                replay += env.run(get_random_action, steps=1)
            else:
                replay += env.run(get_action, steps=1)
        replay = replay[-REPLAY_SIZE:]
        result['rewards'].append(replay.reward()[-1].item())

        if step > UPDATE_START and step % UPDATE_INTERVAL == 0:
            sample = random.sample(replay, BATCH_SIZE)
            batch = ch.ExperienceReplay(sample)

            # Pre-compute some quantities
            masses = actor(batch.state())
            actions = masses.rsample()
            log_probs = masses.log_prob(actions)
            q_values = torch.min(critic_1(batch.state(), actions.detach()),
                                 critic_2(batch.state(), actions.detach())).view(-1, 1)

            # Compute Q losses
            v_next = target_value_critic(batch.next_state()).view(-1, 1)
            q_old_pred1 = critic_1(batch.state(), batch.action().detach()).view(-1, 1)
            q_old_pred2 = critic_2(batch.state(), batch.action().detach()).view(-1, 1)
            qloss1 = ch.algorithms.sac.action_value_loss(q_old_pred1,
                                                         v_next.detach(),
                                                         batch.reward(),
                                                         batch.done(),
                                                         DISCOUNT)
            qloss2 = ch.algorithms.sac.action_value_loss(q_old_pred2,
                                                         v_next.detach(),
                                                         batch.reward(),
                                                         batch.done(),
                                                         DISCOUNT)

            # Update Q-functions by one step of gradient descent
            qloss = qloss1 + qloss2
            critics_optimiser.zero_grad()
            qloss.backward()
            critics_optimiser.step()
            result['qlosses'].append(qloss.item())

            # Update V-function by one step of gradient descent
            v_pred = value_critic(batch.state()).view(-1, 1)
            vloss = ch.algorithms.sac.state_value_loss(v_pred,
                                                       log_probs.detach(),
                                                       q_values.detach(),
                                                       alpha=ENTROPY_WEIGHT)
            value_critic_optimiser.zero_grad()
            vloss.backward()
            value_critic_optimiser.step()
            result['vlosses'].append(vloss.item())

            # Update policy by one step of gradient ascent
            q_actions = critic_1(batch.state(), actions).view(-1, 1)
            policy_loss = ch.algorithms.sac.policy_loss(log_probs,
                                                        q_actions,
                                                        alpha=ENTROPY_WEIGHT)
            actor_optimiser.zero_grad()
            policy_loss.backward()
            actor_optimiser.step()
            result['plosses'].append(policy_loss.item())

            # Update target value network
            ch.models.polyak_average(target_value_critic,
                                     value_critic,
                                     POLYAK_FACTOR)
    result['pweights'] = list(actor.parameters())
    result['vweights'] = list(value_critic.parameters())
    result['vweights_target'] = list(target_value_critic.parameters())
    result['qweights1'] = list(critic_1.parameters())
    result['qweights2'] = list(critic_2.parameters())
    return result


def close(a, b):
    return (a-b).norm(p=2) <= TOL


class TestSpinningUpSAC(unittest.TestCase):

    def setUp(self):
        torch.set_default_tensor_type(torch.DoubleTensor)
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        torch.set_default_tensor_type(torch.FloatTensor)
        torch.set_default_dtype(torch.float32)

    def test_sac(self):
        torch.autograd.set_detect_anomaly(True)
        cherry = train_cherry()
        spinup = train_spinup()

        for key in cherry.keys():
            self.assertTrue(len(cherry[key]) > 0)
            self.assertTrue(len(spinup[key]) > 0)
            for cv, sv in zip(cherry[key], spinup[key]):
                if isinstance(cv, torch.Tensor):
                    self.assertTrue(close(cv, sv))
                else:
                    self.assertTrue(abs(cv - sv) <= TOL)


if __name__ == "__main__":
    unittest.main()
