#!/usr/bin/env python3

import random
import copy
import numpy as np
import gym

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import cherry as ch
from cherry import envs


ACTION_DISCRETISATION = 5
DISCOUNT = 0.99
EPSILON = 0.05
HIDDEN_SIZE = 32
LEARNING_RATE = 0.001
MAX_STEPS = 100000
BATCH_SIZE = 128
REPLAY_SIZE = 100000
TARGET_UPDATE_INTERVAL = 2500
UPDATE_INTERVAL = 1
UPDATE_START = 10000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class DQN(nn.Module):

    def __init__(self, hidden_size, num_actions=5):
        super().__init__()
        layers = [nn.Linear(3, hidden_size),
                  nn.Tanh(),
                  nn.Linear(hidden_size, hidden_size),
                  nn.Tanh(),
                  nn.Linear(hidden_size, num_actions)]
        self.dqn = nn.Sequential(*layers)
        self.egreedy = ch.nn.EpsilonGreedy(EPSILON)

    def forward(self, state):
        values = self.dqn(state)
        action = self.egreedy(values)
        return action, values


def create_target_network(network):
    target_network = copy.deepcopy(network)
    for param in target_network.parameters():
        param.requires_grad = False
    return target_network


def convert_discrete_to_continuous_action(action):
    return action.to(dtype=torch.float32) - ACTION_DISCRETISATION // 2


def main(env):
    env = gym.make(env)
    env.seed(SEED)
    env = envs.Torch(env)
    env = envs.ActionLambda(env, convert_discrete_to_continuous_action)
    env = envs.Logger(env)
    env = envs.Runner(env)

    replay = ch.ExperienceReplay()
    agent = DQN(HIDDEN_SIZE, ACTION_DISCRETISATION)
    target_agent = create_target_network(agent)
    optimiser = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    def get_random_action(state):
        action = torch.tensor([[random.randint(0, ACTION_DISCRETISATION - 1)]])
        return action

    def get_action(state):
        # Original sampling (for unit test)
        #if random.random() < EPSILON:
        #  action = torch.tensor([[random.randint(0, ACTION_DISCRETISATION - 1)]])
        #else:
        #  action = agent(state)[1].argmax(dim=1, keepdim=True)
        #return action
        return agent(state)[0]

    for step in range(1, MAX_STEPS + 1):
        with torch.no_grad():
            if step < UPDATE_START:
                replay += env.run(get_random_action, steps=1)
            else:
                replay += env.run(get_action, steps=1)

            replay = replay[-REPLAY_SIZE:]

        if step > UPDATE_START and step % UPDATE_INTERVAL == 0:
            # Randomly sample a batch of experience
            batch = random.sample(replay, BATCH_SIZE)
            batch = ch.ExperienceReplay(batch)

            # Compute targets
            target_values = target_agent(batch.next_state())[1].max(dim=1, keepdim=True)[0]
            target_values = batch.reward() + DISCOUNT * (1 - batch.done()) * target_values

            # Update Q-function by one step of gradient descent
            pred_values = agent(batch.state())[1].gather(1, batch.action())
            value_loss = F.mse_loss(pred_values, target_values)
            optimiser.zero_grad()
            value_loss.backward()
            optimiser.step()

        if step > UPDATE_START and step % TARGET_UPDATE_INTERVAL == 0:
            # Update target network
            target_agent = create_target_network(agent)


if __name__ == '__main__':
    main('Pendulum-v0')
