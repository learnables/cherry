#!/usr/bin/env python3

"""
TODO:
    * Add clipping objective for the value loss
    * Clean up code to clone a new ExperienceReplay
    * Clean up debugging mess
    * Add replay.myattr as a proxy for replay.info['myattr'] (.totensor() if applicable)
    * Bug in A2C: it seems like I forgot to subtract the value from the advantage.
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
RECORD = False
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
PPO_STEPS = 2048
PPO_BSZ = 64
PPO_CLIP_VALUE = True

OPENAI = True

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
        # Load debug weights
        #weights = th.load('./model.pth')
        #self.actor[0].weight.data.copy_(weights['base.actor.0.weight'])
        #self.actor[0].bias.data.copy_(weights['base.actor.0.bias'])
        #self.actor[2].weight.data.copy_(weights['base.actor.2.weight'])
        #self.actor[2].bias.data.copy_(weights['base.actor.2.bias'])
        #self.actor[4].weight.data.copy_(weights['dist.fc_mean.weight'])
        #self.actor[4].bias.data.copy_(weights['dist.fc_mean.bias'])

        #self.critic[0].weight.data.copy_(weights['base.critic.0.weight'])
        #self.critic[0].bias.data.copy_(weights['base.critic.0.bias'])
        #self.critic[2].weight.data.copy_(weights['base.critic.2.weight'])
        #self.critic[2].bias.data.copy_(weights['base.critic.2.bias'])
        #self.critic[4].weight.data.copy_(weights['base.critic_linear.weight'])
        #self.critic[4].bias.data.copy_(weights['base.critic_linear.bias'])

    def forward(self, x):
        action_scores = self.actor(x)
        action_density = self.action_dist(action_scores)
        value = self.critic(x)
        return action_density, value


def update(replay, optimizer, policy, env, lr_schedule):
    # GAE
    full_rewards = replay.rewards
    values = [info['value'] for info in replay.infos]
    _, next_state_value = policy(replay.next_states[-1])
    values += [next_state_value]
    rewards, returns = ch.rewards.gae(GAMMA,
                                         TAU,
                                         replay.rewards,
                                         replay.dones,
                                         values,
                                         bootstrap=values[-2])
    advantages = [(a - v) for a, v in zip(returns, values)]
    advantages = ch.utils.normalize(ch.utils.totensor(advantages))[0]

    # Somehow create a new replay with updated rewards (elegant)
    new_replay = ch.ExperienceReplay()
    for sars, reward, adv, ret in zip(replay, rewards, advantages, returns):
        sars.reward = reward
        sars.info['advantage'] = adv.detach()
        sars.info['return'] = ret.detach()
        new_replay.add(**sars)
    replay = new_replay

    # Debug stuff
    rs = []
    obs = []
    obc = []
    ls = []
    adv = []
    ent = []
    vl = []
    ret = []
    mean = lambda a: sum(a) / len(a)

    # Perform some optimization steps
    for step in range(PPO_EPOCHS):
        batch_replay = replay.sample(PPO_BSZ)

        # Compute loss
        loss = 0.0
        for transition in batch_replay:

            mass, value = policy(transition.state)
            entropy = mass.entropy().sum(-1)
            log_prob = mass.log_prob(transition.action).sum(-1)

            ratio = th.exp(log_prob - transition.info['log_prob'].detach())
            objective = ratio * transition.info['advantage']
            objective_clipped = ratio.clamp(1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * transition.info['advantage']
            policy_loss = - th.min(objective, objective_clipped)
            value_loss = 0.5 * (transition.info['return'] - value)**2

            if PPO_CLIP_VALUE:
                old_value = transition.info['value'].detach()
                clipped_value = old_value + (value - old_value)
                clipped_value.clamp_(-PPO_CLIP, PPO_CLIP)
                clipped_loss = 0.5 * (transition.info['return'] - clipped_value)**2
                value_loss = th.max(value_loss, clipped_loss)

            loss += policy_loss - ENT_WEIGHT * entropy + V_WEIGHT * value_loss

            ls.append(policy_loss)
            ent.append(entropy)
            vl.append(value_loss)

        # Take optimization step
        loss /= len(batch_replay)
        optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_NORM)
        optimizer.step()

    # Log metrics
    env.log('policy loss', mean(ls).item())
    env.log('policy entropy', mean(ent).item())
    env.log('value loss', mean(vl).item())
    openai = ' openai' if OPENAI else ''
    #ppt.plot(mean(ls).item(), 'policy cherry' + openai)
    #ppt.plot(mean(ent).item(), 'entropy cherry' + openai)
    #ppt.plot(mean(vl).item(), 'value cherry' + openai)
    #ppt.plot(mean(env.all_rewards[-2048:]), 'rewards cherry' + openai)

    # Update the parameters on schedule
    if LINEAR_SCHEDULE:
        lr_schedule.step()


def get_action_value(state, policy):
    mass, value = policy(state)
    action = mass.sample()
    info = {
        'log_prob': mass.log_prob(action).sum(-1),  # Cache log_prob for later
        'value': value
    }
    return action, info


if __name__ == '__main__':
    env_name = 'CartPoleBulletEnv-v0'
    env_name = 'AntBulletEnv-v0'
    env = gym.make(env_name)
    env = envs.AddTimestep(env)
    env = envs.Logger(env, interval=2*PPO_STEPS)
    if OPENAI:
        env = envs.OpenAINormalize(env)
    else:
        env = envs.Normalized(env,
                              normalize_state=True,
                              normalize_reward=False,
                              scale_reward=0.1)
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
    optimizer = optim.Adam(policy.parameters(), lr=LR, eps=1e-5)
    num_updates = TOTAL_STEPS // PPO_STEPS + 1
    lr_schedule = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - epoch/num_updates)
    replay = ch.ExperienceReplay()
    get_action = lambda state: get_action_value(state, policy)

    for epoch in range(num_updates):
        # We use the Runner collector, but could've written our own
        num_samples, num_episodes = env.run(get_action,
                                            replay,
                                            steps=PPO_STEPS,
                                            render=RENDER)

        # Update policy
        update(replay, optimizer, policy, env, lr_schedule)
        replay.empty()

        if RECORD and epoch % 10 == 0:
            record_env.run(get_action, episodes=3, render=True)
