import random
import gym
import numpy as np

from itertools import count

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import cherry as ch
from cherry.envs import TorchEnvWrapper
from cherry.rewards import discount_rewards

from actor_critic_cartpole import finish_episode

SEED = 567
GAMMA = 0.99
RENDER = False
V_WEIGHT = 0.5

random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)


class ActorCriticNet(nn.Module):
    def __init__(self):
        super(ActorCriticNet, self).__init__()
        self.affine1 = nn.Linear(3, 128)
        self.action_head = nn.Linear(128, 1)
        self.value_head = nn.Linear(128, 1)
        self.std = th.tensor(0.01, requires_grad=False)

    def forward(self, x):
        x = th.tanh(self.affine1(x))
        action_scores = th.tanh(self.action_head(x))
        action_mass = Normal(action_scores, self.std)
        value = self.value_head(x)
        return action_mass, value


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    env = TorchEnvWrapper(env)
    env.seed(SEED)

    policy = ActorCriticNet()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    running_reward = 10.0
    replay = ch.ExperienceReplay()

    for i_episode in count(1):
        state = env.reset()
        for _ in range(50):
            for t in range(10000):  # Don't infinite loop while learning
                mass, value = policy(state)
                action = mass.sample()
                old_state = state
                state, reward, done, _ = env.step(action)
                replay.add(old_state, action, reward, state, done, info={
                    'log_prob': mass.log_prob(action),  # Cache log_prob for later
                    'value': value
                })
                if RENDER:
                    env.render()
                if done:
                    break

        episode_reward = discount_rewards(GAMMA, replay.list_rewards, replay.list_dones)
        episode_reward = sum(episode_reward).item()
        running_reward = running_reward * 0.99 + episode_reward * 0.01
        finish_episode(replay, optimizer)
        replay.empty()
        if i_episode % 10 == 0:
            print('Episode {}\tLast rewards: {:2f}\tRunning discounted reward: {:.2f}'.format(
                  i_episode, episode_reward, running_reward))
        if running_reward > 1090.0:
            print('Solved! Running reward is now {} and '
                  'the last episode runs to {} time steps!'.format(running_reward, t))
            break
