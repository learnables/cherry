#!/usr/bin/env python3

import random
import gym
import numpy as np

import torch as th
import torch.nn as nn
import torch.optim as optim

import cherry as ch
import cherry.distributions as distributions
import cherry.envs as envs
from cherry.models import atari
from cherry.algorithms import ppo

RENDER = False
RECORD = True
SEED = 420
TOTAL_STEPS = 10000000
LR = 2.5e-4
GAMMA = 0.99
TAU = 0.95
V_WEIGHT = 0.5
ENT_WEIGHT = 0.01
GRAD_NORM = 0.5
LINEAR_SCHEDULE = True
PPO_CLIP = 0.1
PPO_EPOCHS = 10
PPO_BSZ = 256
PPO_NUM_BATCHES = 4
PPO_STEPS = 1024


class NatureCNN(nn.Module):

    def __init__(self, env, hidden_size=512):
        super(NatureCNN, self).__init__()
        self.input_size = 4
        self.features = atari.NatureFeatures(self.input_size, hidden_size)
        self.critic = atari.NatureCritic(hidden_size)
        self.actor = atari.NatureActor(hidden_size, env.action_size)
        self.action_dist = distributions.ActionDistribution(env,
                                                            use_probs=False)

    def forward(self, x):
        x = x.view(-1, self.input_size, 84, 84).mul(1 / 255.0)
        features = self.features(x)
        value = self.critic(features)
        density = self.actor(features)
        mass = self.action_dist(density)
        return mass, value


def update(replay, optimizer, policy, env, lr_schedule):
    _, next_state_value = policy(replay[-1].next_state)
    # NOTE: Kostrikov uses GAE here.
    advantages = ch.generalized_advantage(GAMMA,
                                          TAU,
                                          replay.reward(),
                                          replay.done(),
                                          replay.value(),
                                          next_state_value)

    advantages = advantages.view(-1, 1)
#    advantages = ch.utils.normalize(advantages, epsilon=1e-5).view(-1, 1)
#    rewards = [a + v for a, v in zip(advantages, replay.value())]
    rewards = advantages + replay.value()

#    rewards = ch.discount(GAMMA,
#                          replay.reward(),
#                          replay.done(),
#                          bootstrap=next_state_value)
#    rewards = rewards.detach()
#    advantages = rewards.detach() - replay.value().detach()
#    advantages = ch.utils.normalize(advantages, epsilon=1e-5).view(-1, 1)

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
        advs = ch.normalize(batch.advantage(), epsilon=1e-8)
        new_log_probs = masses.log_prob(batch.action()).sum(-1, keepdim=True)
        entropy = masses.entropy().sum(-1).mean()
        policy_loss = ppo.policy_loss(new_log_probs,
                                      batch.log_prob(),
#                                      batch.advantage(),
                                      advs,
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

        policy_losses.append(policy_loss.item())
        entropies.append(entropy.item())
        value_losses.append(value_loss.item())

    # Log metrics
    env.log('policy loss', mean(policy_losses))
    env.log('policy entropy', mean(entropies))
    env.log('value loss', mean(value_losses))

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


def main(env='PongNoFrameskip-v4'):
    random.seed(SEED)
    np.random.seed(SEED)
    th.manual_seed(SEED)

    env = gym.make(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = envs.OpenAIAtari(env)
    env = envs.VisdomLogger(env, title='Cherry PPO', interval=PPO_STEPS)
    env = envs.Torch(env)
    env = envs.Runner(env)
    env.seed(SEED)

    policy = NatureCNN(env)
    optimizer = optim.Adam(policy.parameters(), lr=LR, eps=1e-5)
    num_updates = TOTAL_STEPS // PPO_STEPS + 1
    lr_schedule = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - epoch/num_updates)
    get_action = lambda state: get_action_value(state, policy)

    for epoch in range(num_updates):
        policy.cpu()
        replay = env.run(get_action, steps=PPO_STEPS, render=RENDER)
        update(replay, optimizer, policy, env, lr_schedule)


if __name__ == '__main__':
    env_name = 'PongNoFrameskip-v4'
    env_name = 'BreakoutNoFrameskip-v4'
    main(env_name)
