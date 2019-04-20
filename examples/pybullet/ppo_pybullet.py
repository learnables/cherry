#!/usr/bin/env python3

# See: https://github.com/openai/roboschool/issues/15
from OpenGL import GLU
import ppt

import random
import gym
import numpy as np
import pybullet_envs
import roboschool

import torch as th
import torch.nn as nn
import torch.optim as optim

import cherry as ch
import cherry.distributions as dist
import cherry.models as models
import cherry.envs as envs
from cherry.algorithms import ppo

RENDER = False
RECORD = True
SEED = 42
TOTAL_STEPS = 10000000
LR = 3e-4
GAMMA = 0.99
TAU = 0.95
V_WEIGHT = 0.5
ENT_WEIGHT = 0.0
GRAD_NORM = 0.5
LINEAR_SCHEDULE = True
PPO_CLIP = 0.2
PPO_EPOCHS = 10
PPO_BSZ = 64
PPO_NUM_BATCHES = 32
PPO_STEPS = 2048

random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)


class ActorCriticNet(nn.Module):
    def __init__(self, env):
        super(ActorCriticNet, self).__init__()
        self.actor = models.control.Actor(env.state_size,
                                          env.action_size,
                                          layer_sizes=[64, 64])
        self.critic = models.control.ControlMLP(env.state_size, 1)

        self.action_dist = dist.ActionDistribution(env,
                                                   use_probs=False,
                                                   reparam=False)

    def forward(self, x):
        action_scores = self.actor(x)
        action_density = self.action_dist(action_scores)
        value = self.critic(x)
        return action_density, value


def update(replay, optimizer, policy, env, lr_schedule):
    _, next_state_value = policy(replay[-1].next_state)
    advantages = ch.rewards.generalized_advantage(GAMMA,
                                                  TAU,
                                                  replay.reward(),
                                                  replay.done(),
                                                  replay.value(),
                                                  next_state_value)

    advantages = ch.utils.normalize(advantages, epsilon=1e-5).view(-1, 1)
    rewards = [a + v for a, v in zip(advantages, replay.value())]


    for i, sars in enumerate(replay):
        sars.reward = rewards[i].detach()
        sars.advantage = advantages[i].detach()

    # Logging
    policy_losses = []
    entropies = []
    value_losses = []
    mean = lambda a: sum(a) / len(a)

    # Perform some optimization steps
    for step in range(PPO_EPOCHS * PPO_NUM_BATCHES):
        batch = replay.sample(PPO_BSZ)
        masses, values = policy(batch.state())

        # Compute losses
        new_log_probs = masses.log_prob(batch.action()).sum(-1, keepdim=True)
        entropy = masses.entropy().sum(-1).mean()
        policy_loss = ppo.policy_loss(new_log_probs,
                                      batch.log_prob(),
                                      batch.advantage(),
                                      clip=PPO_CLIP)
        value_loss = ppo.state_value_loss(values,
                                          batch.value().detach(),
                                          batch.reward(),
                                          clip=PPO_CLIP)
        loss = policy_loss - ENT_WEIGHT * entropy + V_WEIGHT * value_loss

        # Take optimization step
        optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_NORM)
        optimizer.step()

        policy_losses.append(policy_loss)
        entropies.append(entropy)
        value_losses.append(value_loss)

    # Log metrics
    env.log('policy loss', mean(policy_losses).item())
    env.log('policy entropy', mean(entropies).item())
    env.log('value loss', mean(value_losses).item())
    ppt.plot(mean(env.all_rewards[-10000:]), 'PPO results')

    # Update the parameters on schedule
    if LINEAR_SCHEDULE:
        lr_schedule.step()


def get_action_value(state, policy):
    mass, value = policy(state)
    action = mass.sample()
    info = {
        'log_prob': mass.log_prob(action).sum(-1).detach(),
        'value': value,
    }
    return action, info


if __name__ == '__main__':
    # env_name = 'CartPoleBulletEnv-v0'
    env_name = 'AntBulletEnv-v0'
    env_name = 'MinitaurTrottingEnv-v0'
#    env_name = 'RoboschoolAnt-v1'
    env = gym.make(env_name)
    env = envs.AddTimestep(env)
    env = envs.Logger(env, interval=PPO_STEPS)
    env = envs.Normalizer(env, states=True, rewards=True)
    env = envs.Torch(env)
    env = envs.Runner(env)
    env.seed(SEED)

    th.set_num_threads(1)
    policy = ActorCriticNet(env)
    optimizer = optim.Adam(policy.parameters(), lr=LR, eps=1e-5)
    num_updates = TOTAL_STEPS // PPO_STEPS + 1
    lr_schedule = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - epoch/num_updates)
    get_action = lambda state: get_action_value(state, policy)

    for epoch in range(num_updates):
        # We use the Runner collector, but could've written our own
        replay = env.run(get_action, steps=PPO_STEPS, render=False)

        # Update policy
        update(replay, optimizer, policy, env, lr_schedule)

    mean = lambda x: sum(x) / len(x)
    result = mean(env.all_rewards[-10000:])
    data = {
        'result': result,
        'env': env_name,
        'all_rewards': env.all_rewards,
        'all_dones': env.all_dones,
        'infos': env.values,
    }
    th.save(data, './regression_test/' + env_name + '.pickle')
    th.save(policy.state_dict(),
            './regression_test/' + env_name + '.pth')
