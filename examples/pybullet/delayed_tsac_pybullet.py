#!/usr/bin/env python3

"""
An implementation of Soft Actor-Critic.
"""

try:
    from OpenGL import GLU
    import roboschool
except:
    print('Failed to import OpenGL')
import copy
import random
import numpy as np
import gym
import pybullet_envs
import tqdm

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cherry as ch
import cherry.envs as envs
import cherry.distributions as distributions
from cherry.algorithms import sac
import wandb

SEED = 42
RENDER = False
BATCH_SIZE = 256
TOTAL_STEPS = 1_000_000
MIN_REPLAY = 1000
REPLAY_SIZE = 1000000

DISCOUNT_FACTOR = 0.99
ALL_LR = 3e-4
VF_TARGET_TAU = 5e-3 
USE_AUTOMATIC_ENTROPY_TUNING = True
TARGET_ENTROPY = -6

# Delay policy and target updates
STEP = 0 
DELAY = 5 

random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)


# Critic Network - Q function approximator
class MLP(nn.Module):

    def __init__(self, input_size, output_size, layer_sizes=None, init_w=3e-3):
        super(MLP, self).__init__()
        if layer_sizes is None:
            layer_sizes = [300, 300]
        self.layers = nn.ModuleList()

        in_size = input_size
        for next_size in layer_sizes:
            fc = nn.Linear(in_size, next_size)
            self.layers.append(fc)
            in_size = next_size

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, *args, **kwargs):
        h = th.cat(args, dim=1)
        for fc in self.layers:
            h = F.relu(fc(h))
        output = self.last_fc(h)
        return output


# Actor Network - Parameterized Policy Function
class Policy(MLP):

    def __init__(self, input_size, output_size, layer_sizes=None, init_w=1e-3):
        super(Policy, self).__init__(input_size=input_size,
                                     output_size=output_size,
                                     layer_sizes=layer_sizes,
                                     init_w=init_w)
        features_size = self.layers[-1].weight.size(0)
        self.log_std = nn.Linear(features_size, output_size)
        self.log_std.weight.data.uniform_(-init_w, init_w)
        self.log_std.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        h = state
        for fc in self.layers:
            h = F.relu(fc(h))

        mean = self.last_fc(h)
        log_std = self.log_std(h).clamp(-20.0, 2.0)
        std = log_std.exp()
        density = distributions.TanhNormal(mean, std)
        return density


# Gradient Step - Adopted from [1] Section 6 and [2] Section 5.2
def update(env,
           replay,
           policy,
           critic_qf1,
           critic_qf2,
           target_qf1,
           target_qf2,
           log_alpha,
           policy_optimizer,
           critic_qf1_optimizer,
           critic_qf2_optimizer,
           alpha_optimizer,
           target_entropy,
           step):

    global DELAY, STEP
    STEP += 1

    batch = replay.sample(BATCH_SIZE)
    density = policy(batch.state())
    # NOTE: The following lines are specific to the TanhNormal policy.
    #       Other policies should constrain the output of the policy net.
    actions, log_probs = density.rsample_and_log_prob()
    log_probs = log_probs.sum(dim=1, keepdim=True)

    # Entropy weight loss
    if USE_AUTOMATIC_ENTROPY_TUNING:
        alpha_loss = sac.entropy_weight_loss(log_alpha,
                                             log_probs.detach(),
                                             target_entropy)
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()
        alpha = log_alpha.exp()
    else:
        alpha = th.ones(1)
        alpha_loss = th.zeros(1)


    # QF loss
    qf1_estimate = critic_qf1(batch.state(), batch.action().detach())
    qf2_estimate = critic_qf2(batch.state(), batch.action().detach())


    density = policy(batch.next_state())
    next_actions, next_log_probs = density.rsample_and_log_prob()
    next_log_probs = log_probs.sum(dim=1, keepdim=True)

    target_q_values = th.min(target_qf1(batch.next_state(), next_actions),
                             target_qf2(batch.next_state(), next_actions)) - alpha * next_log_probs

    critic_qf1_loss = sac.action_value_loss(qf1_estimate, 
                                            target_q_values.detach(),
                                            batch.reward(),
                                            batch.done(),
                                            DISCOUNT_FACTOR)

    critic_qf2_loss = sac.action_value_loss(qf2_estimate,
                                            target_q_values.detach(),
                                            batch.reward(),
                                            batch.done(),
                                            DISCOUNT_FACTOR)

    
    # Log debugging values
    wandb.log({
        'alpha Loss': alpha_loss.item(),
        'alpha': alpha.item(),
        "QF1 Loss": critic_qf1_loss.item(),
        "QF2 Loss": critic_qf2_loss.item(),
        "Average Rewards": batch.reward().mean().item(),
    }, step=step)

    # Update Critic Networks
    critic_qf1_optimizer.zero_grad()
    critic_qf1_loss.backward()
    critic_qf1_optimizer.step()

    critic_qf2_optimizer.zero_grad()
    critic_qf2_loss.backward()
    critic_qf2_optimizer.step()

    # Delayed Updates
    if STEP % DELAY == 0:

        # Policy loss
        q_values = th.min(critic_qf1(batch.state(), actions),
                          critic_qf2(batch.state(), actions))
        policy_loss = sac.policy_loss(log_probs, q_values, alpha)

        wandb.log({"Policy Loss": policy_loss.item()}, step=step)

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # Move target approximator parameters towards critic parameters per [3]
        ch.models.polyak_average(source=target_qf1,
                                 target=critic_qf1,
                                 alpha=VF_TARGET_TAU)

        ch.models.polyak_average(source=target_qf2,
                                 target=critic_qf2,
                                 alpha=VF_TARGET_TAU)


def main(env='HalfCheetahBulletEnv-v0'):
    random.seed(SEED)
    np.random.seed(SEED)
    th.manual_seed(SEED)
    th.set_num_threads(2)
    env = gym.make(env)
    env = envs.ActionSpaceScaler(env)
    env = envs.Torch(env)
    env = envs.Runner(env)
    env.seed(SEED)

    log_alpha = th.zeros(1, requires_grad=True)
    if USE_AUTOMATIC_ENTROPY_TUNING:
        # Heuristic target entropy
        target_entropy = -np.prod(env.action_space.shape).item()
    else:
        target_entropy = TARGET_ENTROPY

    state_size = env.state_size
    action_size = env.action_size

    policy = Policy(input_size=state_size, output_size=action_size)
    critic_qf1 = MLP(input_size=state_size+action_size, output_size=1)
    critic_qf2 = MLP(input_size=state_size+action_size, output_size=1)
    target_qf1 = copy.deepcopy(critic_qf1)
    target_qf2 = copy.deepcopy(critic_qf2)

    policy_opt = optim.Adam(policy.parameters(), lr=ALL_LR)
    qf1_opt = optim.Adam(critic_qf1.parameters(), lr=ALL_LR)
    qf2_opt = optim.Adam(critic_qf2.parameters(), lr=ALL_LR)
    alpha_opt = optim.Adam([log_alpha], lr=ALL_LR)

    replay = ch.ExperienceReplay()
    get_action = lambda state: policy(state).rsample()

    episode_reward = 0.0
    for step in tqdm.trange(TOTAL_STEPS):
        # Collect next step
        ep_replay = env.run(get_action, steps=1, render=RENDER)
        episode_reward += ep_replay[-1].reward
        if ep_replay[-1].done:
            wandb.log({
                'Episode Rewards': episode_reward,
            }, step=step)
            episode_reward = 0.0

        # Update policy
        replay += ep_replay
        replay = replay[-REPLAY_SIZE:]
        if len(replay) > MIN_REPLAY:
            update(env,
                   replay,
                   policy,
                   critic_qf1,
                   critic_qf2,
                   target_qf1,
                   target_qf2,
                   log_alpha,
                   policy_opt,
                   qf1_opt,
                   qf2_opt,
                   alpha_opt,
                   target_entropy,
                   step,
            )

if __name__ == '__main__':
    env_name = 'CartPoleBulletEnv-v0'
    env_name = 'AntBulletEnv-v0'
    env_name = 'HalfCheetahBulletEnv-v0'
    #env_name = 'MinitaurTrottingEnv-v0'
    env_name = 'RoboschoolAtlasForwardWalk-v1'
    env_name = 'Ant-v3'
    #env_name = 'HalfCheetah-v3'
    wandb.init(
        project='qmcrl',
        name='baseline-' + env_name,
    )
    main(env_name)
