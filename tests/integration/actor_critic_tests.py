#!/usr/bin/env python3 -W ignore::DeprecationWarning

import unittest
import gym
import random
import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cherry.envs as envs
from cherry.rewards import discount
from cherry.utils import normalize
import cherry.distributions as distributions

SEED = 567
GAMMA = 0.99
RENDER = False
V_WEIGHT = 0.5

# NOTE: those values are from Seb's notebook.
GROUND_TRUTHS = [
    10.417155693377921,
    12.215221906973023,
    14.607232694695206,
    18.908556685680427,
    22.87979520858179,
    24.29453858531894,
    37.676546748954266,
    46.41569497544136,
    57.99665823088736,
    71.30742311266546,
    81.77514934640226,
    89.51613224152003,
    99.72494989788753,
    108.95677039052104,
    117.66213509027638,
    125.5351108811495,
    132.40108942305517,
    136.95963264101184,
    138.35750805596098,
    135.74780181088178,
    137.6316201664582,
    140.29016195280857,
    145.70546276844132,
    150.89699375688014,
    155.278489324677,
    159.0471577959876,
    162.96298359002714,
    166.50438624701357,
    169.7071673303807,
    172.60370513235517,
    175.22328200004628,
    177.59238036329378,
    179.7349504569476,
    179.96087035427803,
    181.87697034963148,
    183.5504568393536,
    185.1233280234277,
    185.7736045286022,
    187.13390294368133,
    188.3641324469418,
    189.47672995783756,
    188.6863432033911,
    188.9770730695244,
    186.67878307683145,
    183.5794269112394,
    185.02034377993306,
    182.5042251947773,
    182.681434641144,
    183.96950161491378,
    185.5023046070704,
]


class ActorCriticNet(nn.Module):

    def __init__(self, env):
        super(ActorCriticNet, self).__init__()
        self.affine = nn.Linear(env.state_size, 128)
        self.action_head = nn.Linear(128, env.action_size)
        self.value_head = nn.Linear(128, 1)
        self.distribution = distributions.ActionDistribution(env,
                                                             use_probs=True)

    def forward(self, x):
        x = F.relu(self.affine(x))
        action_scores = self.action_head(x)
        action_mass = self.distribution(F.softmax(action_scores, dim=1))
        value = self.value_head(x)
        return action_mass, value


def update(replay, optimizer):
    policy_loss = []
    value_loss = []
    rewards = discount(GAMMA, replay.rewards, replay.dones)
    rewards = normalize(rewards)
    for info, reward in zip(replay.infos, rewards):
        log_prob = info['log_prob']
        value = info['value']
        policy_loss.append(-log_prob * (reward - value.item()))
        value_loss.append(F.mse_loss(value, reward.detach()))
    optimizer.zero_grad()
    loss = th.stack(policy_loss).sum() + V_WEIGHT * th.stack(value_loss).sum()
    loss.backward()
    optimizer.step()


def get_action_value(state, policy):
    mass, value = policy(state)
    action = mass.sample()
    info = {
        'log_prob': mass.log_prob(action),
        'value': value
    }
    return action, info


class TestActorCritic(unittest.TestCase):

    def test_training(self):
        """
        Issue: Depending on the computer architecture,
        PyTorch will represent floating numbers differently differently.
        For example, the above is the output from Seb's MacBook, but it doesn't
        exactly match the output on his desktop after episode 109.
        Saving weights / initializing using numpy didn't work either.
        Is there a workaround ?

        To be more specific, it seems to be the way PyTorch stores FP, since
        when calling .tolist() on the weights, all decimals match.
        Or, it's an out-of-order execution issue. (We did try to use a single
        MKL/OMP thread.)
        """
        th.set_num_threads(1)
        random.seed(SEED)
        np.random.seed(SEED)
        th.manual_seed(SEED)

        env = gym.make('CartPole-v0')
        env.seed(SEED)
        env = envs.Torch(env)
        env = envs.Runner(env)

        policy = ActorCriticNet(env)
        optimizer = optim.Adam(policy.parameters(), lr=1e-2)
        running_reward = 10.0
        get_action = lambda state: get_action_value(state, policy)

        best_running = 0.0
        for episode in range(0, 100):  # >100 breaks at episode 109
            replay = env.run(get_action, episodes=1)
            update(replay, optimizer)
            running_reward = running_reward * 0.99 + len(replay) * 0.01
            if running_reward >= best_running:
                best_running = running_reward
            if (episode+1) % 10 == 0 and episode < 185:
#                print(running_reward)
                self.assertTrue((GROUND_TRUTHS[episode // 10] - running_reward)**2 <= 1e-8)
#        self.assertTrue(best_running >= 150.0)


if __name__ == '__main__':
    unittest.main()
