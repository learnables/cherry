#!/usr/bin/env python3

"""
An implementation of SoftActorCritic.

TODO:
    * Decide whether to keep rsample_and_log_prob() or change
      to a 3.0*Tanh before inputing to TanhNormal().
      (The latter might actually work.)
"""

import ppt
import copy
import random
import numpy as np
import gym
import pybullet_envs

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cherry as ch
import cherry.envs as envs
import cherry.distributions as distributions
from cherry.algorithms import sac

SEED = 42
RENDER = False
GAMMA = 0.99
BSZ = 128
TOTAL_STEPS = 1000000
MIN_REPLAY = 1000
REPLAY_SIZE = 1000000
ALL_LR = 3e-4
MEAN_REG_WEIGHT = 1e-3
STD_REG_WEIGHT = 1e-3
VF_TARGET_TAU = 0.001
USE_AUTOMATIC_ENTROPY_TUNING = True
TARGET_ENTROPY = -6

random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)


class MLP(nn.Module):

    def __init__(self, input_size, output_size, layer_sizes=None, init_w=3e-3):
        super(MLP, self).__init__()
        if layer_sizes is None:
            layer_sizes = [300, 300]
        self.layers = nn.ModuleList()

        in_size = input_size
        for next_size in layer_sizes:
            fc = nn.Linear(in_size, next_size)
            ch.nn.init.pong_control_(fc)
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


def update(replay,
           policy,
           qf,
           vf,
           target_vf,
           log_alpha,
           policy_opt,
           qf_opt,
           vf_opt,
           alpha_opt,
           target_entropy):

    batch = replay.sample(BSZ)
    density = policy(batch.states)
    # NOTE: The following lines are specific to the TanhNormal policy.
    #       Other policies should constrain the output of the policy net.
    actions, log_probs = density.rsample_and_log_prob()
    log_probs = log_probs.sum(dim=1, keepdim=True)
    policy_mean = density.normal.mean
    policy_log_std = density.normal.stddev.log()

    # Entropy weight loss
    if USE_AUTOMATIC_ENTROPY_TUNING:
        alpha_loss = sac.entropy_weight_loss(log_alpha,
                                             log_probs,
                                             target_entropy)
        alpha_opt.zero_grad()
        alpha_loss.backward()
        alpha_opt.step()
        alpha = log_alpha.exp()
    else:
        alpha = th.ones(1)
        alpha_loss = th.zeros(1)

    # QF loss
    q_old_pred = qf(batch.states, batch.actions.detach())
    v_next = target_vf(batch.next_states)
    qf_loss = sac.q_loss(q_old_pred, v_next, batch.rewards, batch.dones, GAMMA)

    # VF loss
    v_pred = vf(batch.states)
    q_values = qf(batch.states, actions)
    vf_loss = sac.v_loss(v_pred, log_probs, q_values, alpha)

    # Policy loss
    policy_loss = sac.policy_loss(log_probs, q_values, alpha)
    mean_reg_loss = MEAN_REG_WEIGHT * policy_mean.pow(2).mean()
    std_reg_loss = STD_REG_WEIGHT * policy_log_std.pow(2).mean()
    policy_reg_loss = mean_reg_loss + std_reg_loss
    policy_loss += policy_reg_loss

    # Log debugging values
    env.log('alpha Loss:', alpha_loss.item())
    env.log('alpha: ', alpha.item())
    env.log("QF Loss: ", qf_loss.item())
    env.log("VF Loss: ", vf_loss.item())
    env.log("Policy Loss: ", policy_loss.item())
    env.log("Average Rewards: ", batch.rewards.mean().item())
    ppt.plot(replay.rewards[-1000:].mean().item(), 'cherry true rewards')

    # Update
    qf_opt.zero_grad()
    qf_loss.backward()
    qf_opt.step()

    vf_opt.zero_grad()
    vf_loss.backward()
    vf_opt.step()

    policy_opt.zero_grad()
    policy_loss.backward()
    policy_opt.step()

    ch.utils.polyak_average(source=target_vf,
                            target=vf,
                            alpha=VF_TARGET_TAU)


if __name__ == '__main__':
    random.seed(SEED)
    np.random.seed(SEED)
    th.manual_seed(SEED)
    env = gym.make('HalfCheetahBulletEnv-v0')
    env = envs.Logger(env, interval=1000)
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
    qf = MLP(input_size=state_size+action_size, output_size=1)
    vf = MLP(input_size=state_size, output_size=1)
    target_vf = copy.deepcopy(vf)

    policy_opt = optim.Adam(policy.parameters(), lr=ALL_LR)
    qf_opt = optim.Adam(qf.parameters(), lr=ALL_LR)
    vf_opt = optim.Adam(vf.parameters(), lr=ALL_LR)
    alpha_opt = optim.Adam([log_alpha], lr=ALL_LR)

    replay = ch.ExperienceReplay()
    get_action = lambda state: policy(state).rsample()

    for step in range(TOTAL_STEPS):
        # Collect next step
        ep_replay = env.run(get_action, steps=1, render=RENDER)

        # Update policy
        replay += ep_replay
        replay = replay[-REPLAY_SIZE:]
        if len(replay) > MIN_REPLAY:
            update(replay, policy, qf, vf, target_vf, log_alpha, policy_opt,
                   qf_opt, vf_opt, alpha_opt, target_entropy)
