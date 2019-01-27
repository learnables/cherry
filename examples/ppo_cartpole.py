#!/usr/bin/env python3

"""
Simple example of using cherry to solve cartpole with an actor-critic.

The code is an adaptation of the PyTorch reinforcement learning example.
"""

import random
import gym
import numpy as np
import pybullet_envs

import torch as th
import torch.nn as nn
import torch.optim as optim

import cherry as ch
import cherry.policies as policies
import cherry.envs as envs

RENDER = False
SEED = 1
GAMMA = 0.99
TAU = 0.95
V_WEIGHT = 0.5
ENT_WEIGHT = 0.01
GRAD_NORM = 0.5
PPO_CLIP = 0.1
PPO_EPOCHS = 4
PPO_STEPS = 256
PPO_BSZ = 64

random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)


def ikostrikov_init(module, gain=None):
    if gain is None:
        gain = np.sqrt(2.0)
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0.0)
    return module


class ActorCriticNet(nn.Module):
    def __init__(self, env):
        super(ActorCriticNet, self).__init__()
        self.actor = nn.Sequential(
            ikostrikov_init(nn.Linear(env.state_size, 64)),
            nn.Tanh(),
            ikostrikov_init(nn.Linear(64, 64)),
            nn.Tanh(),
            ikostrikov_init(nn.Linear(64, env.action_size), gain=1.0),
        )

        self.critic = nn.Sequential(
            ikostrikov_init(nn.Linear(env.state_size, 64)),
            nn.Tanh(),
            ikostrikov_init(nn.Linear(64, 64)),
            nn.Tanh(),
            ikostrikov_init(nn.Linear(64, 1)),
        )

        self.action_dist = policies.ActionDistribution(env, use_probs=False)

    def forward(self, x):
        action_scores = self.actor(x)
        action_density = self.action_dist(action_scores)
        value = self.critic(x)
        return action_density, value


def update(replay, optimizer, policy, env):

    # GAE
    rewards = replay.rewards
    values = [info['value'] for info in replay.infos]
    _, next_state_value = policy(replay.next_states[-1])
    values += [next_state_value]
    rewards, advantages = ch.rewards.gae(GAMMA,
                                         TAU,
                                         rewards,
                                         replay.dones,
                                         values,
                                         bootstrap=values[-2])

    # Somehow create a new replay with updated rewards (elegant)
    new_replay = ch.ExperienceReplay()
    for sars, reward, adv in zip(replay, rewards, advantages):
        sars.reward = reward
        sars.info['advantage'] = adv
        new_replay.add(**sars)
    replay = new_replay

    # Perform some optimization steps
    for step in range(PPO_EPOCHS):
        batch_replay = replay.sample(PPO_BSZ)

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
        th.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_NORM)
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
    env = gym.make('CartPole-v1')
#    env = gym.make('AntBulletEnv-v0')
    env = envs.Logger(env, interval=PPO_STEPS)
    env = envs.Torch(env)
    env = envs.Runner(env)
    env.seed(SEED)

    policy = ActorCriticNet(env)
    optimizer = optim.RMSprop(policy.parameters(), lr=2.5e-4, eps=1e-5, alpha=0.99)
    replay = ch.ExperienceReplay()
    get_action = lambda state: get_action_value(state, policy)

    episode = 0
    while True:
        # We use the Runner collector, but could've written our own
        num_samples, num_episodes = env.run(get_action, replay, steps=PPO_STEPS, render=RENDER)

        # Update policy
        update(replay, optimizer, policy, env)
        replay.empty()
