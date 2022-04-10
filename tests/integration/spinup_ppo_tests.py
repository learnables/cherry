#!/usr/bin/env python3

import unittest

import torch
import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
import gym
from torch import optim
from torch import nn
from torch.distributions import Normal
import cherry as ch
from cherry import pg
from cherry import td
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
MAX_STEPS = 20000
CHERRY_MAX_STEPS = MAX_STEPS // 200
BATCH_SIZE = 2048
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


class ActorCritic(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.actor = Actor(hidden_size, stochastic=True)
        self.critic = Critic(hidden_size)

    def forward(self, state):
        policy = Normal(self.actor(state), self.actor.policy_log_std.exp())
        value = self.critic(state)
        return policy, value


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

    env = Env()
    agent = ActorCritic(HIDDEN_SIZE)
    actor_optimiser = optim.Adam(agent.actor.parameters(), lr=LEARNING_RATE)
    critic_optimiser = optim.Adam(agent.critic.parameters(), lr=LEARNING_RATE)


    result = {
        'rewards': [],
        'policy_losses': [],
        'value_losses': [],
        'weights': [],
    }

    state, done, total_reward, D = env.reset(), False, 0, []

    for step in range(1, MAX_STEPS + 1):
        # Collect set of trajectories D by running policy π in the environment
        policy, value = agent(state)
        action = policy.sample()
        log_prob_action = policy.log_prob(action)
        next_state, reward, done = env.step(action)
        total_reward += reward
        D.append({'state': state,
                  'action': action,
                  'reward': torch.tensor([reward]),
#                  'done': torch.tensor([done], dtype=torch.float32),
                  'done': torch.tensor([done], dtype=torch.float64),
                  'log_prob_action': log_prob_action,
                  'old_log_prob_action': log_prob_action.detach(),
                  'value': value})
        state = next_state
        result['rewards'].append(reward)
        if done:
            state, total_reward = env.reset(), 0

            if len(D) >= BATCH_SIZE:
                # Compute rewards-to-go R and advantage estimates based on the current value function V
                with torch.no_grad():
                    reward_to_go, advantage, next_value = torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.])
                    for transition in reversed(D):
                        reward_to_go = transition['reward'] + (1 - transition['done']) * (DISCOUNT * reward_to_go)
                        transition['reward_to_go'] = reward_to_go
                        td_error = transition['reward'] + (1 - transition['done']) * DISCOUNT * next_value - transition['value']
                        advantage = td_error + (1 - transition['done']) * DISCOUNT * TRACE_DECAY * advantage
                        transition['advantage'] = advantage
                        next_value = transition['value']
                # Turn trajectories into a single batch for efficiency (valid for feedforward networks)
                trajectories = {k: torch.cat([trajectory[k] for trajectory in D], dim=0) for k in D[0].keys()}
                # Extra step: normalise advantages
                trajectories['advantage'] = (trajectories['advantage'] - trajectories['advantage'].mean()) / (trajectories['advantage'].std() + 1e-8)
                D = []

                for epoch in range(PPO_EPOCHS):
                    # Recalculate outputs for subsequent iterations
                    if epoch > 0:
                        policy, trajectories['value'] = agent(trajectories['state'])
                        trajectories['log_prob_action'] = policy.log_prob(trajectories['action'].detach())

                    # Update the policy by maximising the PPO-Clip objective
                    policy_ratio = (trajectories['log_prob_action'].sum(dim=1) - trajectories['old_log_prob_action'].sum(dim=1)).exp()
                    policy_loss = -torch.min(policy_ratio * trajectories['advantage'], torch.clamp(policy_ratio, min=1 - PPO_CLIP_RATIO, max=1 + PPO_CLIP_RATIO) * trajectories['advantage']).mean()
                    actor_optimiser.zero_grad()
                    policy_loss.backward()
                    actor_optimiser.step()
                    result['policy_losses'].append(policy_loss.item())

                    # Fit value function by regression on mean-squared error
                    value_loss = (trajectories['value'] - trajectories['reward_to_go']).pow(2).mean()
                    critic_optimiser.zero_grad()
                    value_loss.backward()
                    critic_optimiser.step()
                    result['value_losses'].append(value_loss.item())
    result['weights'] = list(agent.parameters())
    return result


def train_cherry():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make('Pendulum-v1')
    env.seed(SEED)
    env = envs.Torch(env)
    env = envs.Runner(env)
    replay = ch.ExperienceReplay()

    agent = ActorCritic(HIDDEN_SIZE)
    actor_optimiser = optim.Adam(agent.actor.parameters(), lr=LEARNING_RATE)
    critic_optimiser = optim.Adam(agent.critic.parameters(), lr=LEARNING_RATE)

    def get_action(state):
            mass, value = agent(state)
            action = mass.sample()
            log_prob = mass.log_prob(action)
            return action, {
                    'log_prob': log_prob,
                    'value': value,
            }

    result = {
        'rewards': [],
        'policy_losses': [],
        'value_losses': [],
        'weights': [],
    }

    for step in range(1, CHERRY_MAX_STEPS + 1):
        replay += env.run(get_action, episodes=1)

        if len(replay) >= BATCH_SIZE:
            for r in replay.reward():
                result['rewards'].append(r.item())
            with torch.no_grad():
                advantages = pg.generalized_advantage(DISCOUNT,
                                                      TRACE_DECAY,
                                                      replay.reward(),
                                                      replay.done(),
                                                      replay.value(),
                                                      torch.zeros(1))
                advantages = ch.normalize(advantages, epsilon=1e-8)
                returns = td.discount(DISCOUNT,
                                      replay.reward(),
                                      replay.done())
                old_log_probs = replay.log_prob()

            new_values = replay.value()
            new_log_probs = replay.log_prob()
            for epoch in range(PPO_EPOCHS):
                # Recalculate outputs for subsequent iterations
                if epoch > 0:
                    masses, new_values = agent(replay.state())
                    new_log_probs = masses.log_prob(replay.action())
                    new_values = new_values.view(-1, 1)

                # Update the policy by maximising the PPO-Clip objective
                policy_loss = ch.algorithms.ppo.policy_loss(new_log_probs,
                                                            old_log_probs,
                                                            advantages,
                                                            clip=PPO_CLIP_RATIO)
                actor_optimiser.zero_grad()
                policy_loss.backward()
                actor_optimiser.step()
                result['policy_losses'].append(policy_loss.item())

                # Fit value function by regression on mean-squared error
                value_loss = ch.algorithms.a2c.state_value_loss(new_values, returns)
                critic_optimiser.zero_grad()
                value_loss.backward()
                critic_optimiser.step()
                result['value_losses'].append(value_loss.item())
            replay.empty()

    result['weights'] = list(agent.parameters())
    return result


def close(a, b):
    return (a-b).norm(p=2) <= TOL


class TestSpinningUpPPO(unittest.TestCase):

    def setUp(self):
        torch.set_default_tensor_type(torch.DoubleTensor)
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        torch.set_default_tensor_type(torch.FloatTensor)
        torch.set_default_dtype(torch.float32)

    def test_ppo(self):
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
