#!/usr/bin/env python3

"""
TODO:
    * Add clipping objective for the value loss
    * Clean up code to clone a new ExperienceReplay
    * Clean up debugging mess
    * Maybe worth it to have cherry.models that defines commonly used
      architectures/init/etc for: Atari, Control, ... ?
"""

import ppt

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
RECORD = True
SEED = 42
TOTAL_UPDATES = 10000000
NUM_UPDATES = 0
LR = 2.5e-4
GAMMA = 0.99
TAU = 0.95
V_WEIGHT = 0.5
ENT_WEIGHT = 0.01
GRAD_NORM = 0.5
LINEAR_SCHEDULE = False
PPO_CLIP = 0.1
PPO_EPOCHS = 4
PPO_STEPS = 2048
PPO_BSZ = 256
PPO_CLIP_VALUE = True
PPO_SCHEDULE_CLIP = True

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

        self.action_dist = policies.ActionDistribution(env,
                                                       use_probs=False,
                                                       reparam=False)

    def forward(self, x):
        action_scores = self.actor(x)
        action_density = self.action_dist(action_scores)
        value = self.critic(x)
        return action_density, value


def update(replay, optimizer, policy, env, lr_schedule):
    global PPO_CLIP, NUM_UPDATES

    # GAE
    full_rewards = rewards = replay.rewards
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
        vl = []
        mean = lambda a: sum(a) / len(a)
        # Compute loss
        loss = 0.0
        for transition in batch_replay:
            mass, value = policy(transition.state)
            log_prob = mass.log_prob(transition.action).sum(-1)
            ratio = th.exp(log_prob - transition.info['log_prob'].detach())
            objective = ratio * transition.info['advantage']
            objective_clipped = ratio.clamp(1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * transition.info['advantage']
            entropy = mass.entropy().sum(-1)
            loss -= th.min(objective, objective_clipped) + ENT_WEIGHT * entropy
            value_loss = (transition.reward - value)**2
            rs.append(ratio)
            obs.append(objective)
            obc.append(objective_clipped)
            ls.append(loss)
            ent.append(entropy)
            adv.append(transition.info['advantage'])
            vl.append(value_loss)
            loss = loss + V_WEIGHT * value_loss
        env.log('policy loss', mean(ls).item())
        env.log('policy entropy', mean(ent).item())
        env.log('ppo clip', PPO_CLIP)
        ppt.plot(mean(ent).item(), 'entropy')
        ppt.plot(mean(ls).item(), 'policy loss')
        ppt.plot(mean(vl).item(), 'value loss')
        ppt.plot(mean(full_rewards).item(), 'rewards')

        # Take optimization step
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        th.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_NORM)
        optimizer.step()

    # Update the parameters on schedule
    NUM_UPDATES += 1
    if LINEAR_SCHEDULE:
        lr_schedule.step()
    if PPO_SCHEDULE_CLIP:
        PPO_CLIP = PPO_CLIP * (1.0 - NUM_UPDATES / TOTAL_UPDATES)


def get_action_value(state, policy):
    mass, value = policy(state)
    action = mass.sample()
    info = {
        'log_prob': mass.log_prob(action).sum(-1),  # Cache log_prob for later
        'value': value
    }
    return action, info


if __name__ == '__main__':
    env_name = 'CartPole-v1'
    env_name = 'AntBulletEnv-v0'
    env = gym.make(env_name)
    env = envs.AddTimestep(env)
    env = envs.Logger(env, interval=2*PPO_STEPS)
    env = envs.Torch(env)
    env = envs.Runner(env)
    env.seed(SEED)

    if RECORD:
        record_env = gym.make(env_name)
        record_env = envs.AddTimestep(record_env)
        record_env = envs.Monitor(record_env, './videos/')
        record_env = envs.Torch(record_env)
        record_env = envs.Runner(record_env)
        record_env.seed(SEED)

    policy = ActorCriticNet(env)
    optimizer = optim.RMSprop(policy.parameters(),
                              lr=LR, eps=1e-5, alpha=0.99)
    lr_schedule = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: LR / (epoch + 1))
    replay = ch.ExperienceReplay()
    get_action = lambda state: get_action_value(state, policy)

    total_steps = 0
    while total_steps < TOTAL_UPDATES:
        # We use the Runner collector, but could've written our own
        num_samples, num_episodes = env.run(get_action,
                                            replay,
                                            steps=PPO_STEPS,
                                            render=RENDER)

        # Update policy
        update(replay, optimizer, policy, env, lr_schedule)
        replay.empty()
        total_steps += num_samples

        if RECORD and (total_steps / PPO_STEPS) % 10 == 0:
            print('Recording')
            record_env.run(get_action, episodes=3, render=True)
