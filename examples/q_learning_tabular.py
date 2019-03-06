#!/usr/bin/env python3

import gym

import torch.nn as nn
import torch.optim as optim

import cherry as ch
import cherry.envs as envs
from cherry.models.tabular import ActionValueFunction
from cherry.distributions import EpsilonGreedy


class Agent(nn.Module):

    def __init__(self, env):
        super(Agent, self).__init__()
        self.env = env
        self.qf = ActionValueFunction(env.state_size,
                                      env.action_size)
        self.qf.values.data.mul_(0.0)
        self.e_greedy = EpsilonGreedy(0.51)

    def forward(self, x):
        x = ch.utils.onehot(x, self.env.state_size)
        q_values = self.qf(x)
        action = self.e_greedy(q_values)
        info = {
            'q_values': q_values,
            'action': action
        }
        return action, info


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
#    env = gym.make('NChain-v0')
    env = envs.Logger(env, interval=1000)
    env = envs.Torch(env)
    env = envs.Runner(env)
    agent = Agent(env)
    discount = 0.99
    epsilon = 1.0
    optimizer = optim.SGD(agent.parameters(), lr=0.0051, momentum=0.0)
    for t in range(1, 100000):
        transition = env.run(agent, steps=1)[0]

        curr_q = transition.info['q_values'][:, transition.action]
        next_state = ch.utils.onehot(transition.next_state.long(),
                                     dim=env.state_size)
        next_q = agent.qf(next_state).max().detach()
        td_error = transition.reward + discount * next_q - curr_q
        curr_q.data.add_(-0.01, td_error)

#        optimizer.zero_grad()
#        loss = td_error.pow(2).mul(0.5)
#        loss.backward()
#        optimizer.step()
        agent.e_greedy.epsilon = epsilon / t
        print(agent.qf.values.norm(p=2).item())
