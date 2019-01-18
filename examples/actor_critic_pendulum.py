import random
import gym
import numpy as np

from itertools import count

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import cherry as ch
import cherry.envs as envs
from cherry.rewards import discount_rewards

from actor_critic_cartpole import update, get_action_value

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
    logger = envs.Logger(env, interval=1000)
    env = envs.Normalized(logger, normalize_reward=True)
    env = envs.Torch(env)
    env = envs.Runner(env)
    env.seed(SEED)

    policy = ActorCriticNet()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    running_reward = 10.0
    replay = ch.ExperienceReplay()

    get_action = lambda state: get_action_value(state, policy)
    for episode in count(1):
        # Sample transitions
        num_samples, num_episodes = env.run(get_action, replay, episodes=1)

        # Update policy
        update(replay, optimizer)
        replay.empty()

        # Compute termination criterion
        # Note: we use the logger's rewards because of normalization
        episode_reward = discount_rewards(GAMMA,
                                          logger.all_rewards[-num_samples:],
                                          logger.all_dones[-num_samples:])
        episode_reward = sum(episode_reward).item()
        running_reward = running_reward * 0.99 + episode_reward * 0.01
        if running_reward > -400.0:
            print('Solved! Running reward is now {} and '
                  'the last episode runs to {} time steps!'.format(running_reward, t))
            break
