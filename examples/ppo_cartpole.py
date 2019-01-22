#!/usr/bin/env python3

"""
Simple example of using cherry to solve cartpole with an actor-critic.

The code is an adaptation of the PyTorch reinforcement learning example.
"""

import random
import gym
import numpy as np

import torch as th
import torch.nn as nn
import torch.optim as optim

import cherry as ch
import cherry.policies as policies
import cherry.envs as envs
from cherry.rewards import discount_rewards
from cherry.utils import normalize

SEED = 567
GAMMA = 0.99
RENDER = False
V_WEIGHT = 0.5
ENT_WEIGHT = 0.01
PPO_CLIP = 0.2

random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)


class ActorCriticNet(nn.Module):
    def __init__(self, env):
        super(ActorCriticNet, self).__init__()
        self.affine1 = nn.Linear(env.state_size, 128)
        self.action_head = nn.Linear(128, env.action_size)
        self.value_head = nn.Linear(128, 1)
        self.action = policies.ActionDistribution(env)

    def forward(self, x):
        x = th.tanh(self.affine1(x))
        action_scores = self.action_head(x)
        action_mass = self.action(action_scores)
        value = self.value_head(x)
        return action_mass, value


def update(replay, optimizer, policy, env):
    policy_loss = []
    value_loss = []

    # Discount and normalize rewards
    rewards = discount_rewards(GAMMA, replay.rewards, replay.dones)
    rewards = normalize(th.tensor(rewards))

    # Somehow create a new replay with updated rewards (elegant)
    new_replay = ch.ExperienceReplay()
    for sars, reward in zip(replay, rewards):
        sars.reward = reward
        sars.info['advantage'] = reward - sars.info['value']
        new_replay.add(**sars)
    replay = new_replay

    # Perform some optimization steps
    for step in range(10):
        batch_replay = replay.sample(32)

        # Debug stuff
        rs = []
        obs = []
        obc = []
        ls = []
        adv = []
        ent = []
        mean = lambda a: sum(a) / len(a)
        # Compute loss
        loss = 0.0
        for transition in batch_replay:
            mass, value = policy(transition.state)
            ratio = th.exp(mass.log_prob(transition.action) - transition.info['log_prob'].detach())
            objective = ratio * transition.info['advantage']
            objective_clipped = ratio.clamp(1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * transition.info['advantage']
            # TODO: Also compute value loss
            loss -= th.min(objective, objective_clipped) + ENT_WEIGHT * mass.entropy()
            rs.append(ratio)
            obs.append(objective)
            obc.append(objective_clipped)
            ls.append(loss)
            ent.append(mass.entropy())
            adv.append(transition.info['advantage'])
            loss += V_WEIGHT * (transition.reward - value)**2
        env.log('policy loss', mean(ls).item())
        env.log('policy entropy', mean(ent).item())

        # Take optimization step
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        th.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()


def get_action_value(state, policy):
    mass, value = policy(state)
    action = mass.sample()
    info = {
        'log_prob': mass.log_prob(action),  # Cache log_prob for later
        'value': value
    }
    return action, info


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = envs.Logger(env, interval=2048)
    env = envs.Torch(env)
    env = envs.Runner(env)
    env.seed(SEED)

    policy = ActorCriticNet(env)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    running_reward = 10.0
    replay = ch.ExperienceReplay()
    get_action = lambda state: get_action_value(state, policy)

    episode = 0
    while True:
        # We use the Runner collector, but could've written our own
        num_samples, num_episodes = env.run(get_action, replay, steps=2048)

        # Update policy
        update(replay, optimizer, policy, env)
        replay.empty()

        # Compute termination criterion
        running_reward = running_reward * 0.99 + num_samples/num_episodes * 0.01
        episode += num_episodes
        # if episode % 10 == 0 or True:
        #     # Should start with 10.41, 12.21, 14.60, then 100:71.30, 200:135.74
        #     print(episode, running_reward)
        # if running_reward > env.spec.reward_threshold:
        #     print('Solved! Running reward now {} and '
        #           'the last episode runs to {} time steps!'.format(running_reward, num_samples))
        #     break
