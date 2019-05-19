#!/usr/bin/env python

"""
This example showcases how to use cherry to design flexible agents for the
simple_rl framework.
It is largely inspired by the simple_example here:
https://github.com/david-abel/simple_rl/blob/master/examples/simple_example.py

Note: Hyperparameters were not tuned. In particular, the CherryQAgent class
uses a couple of constants that seemed to work reasonably well across
different architectures.

Note: It is recommended to have a look at the Q-learning implementation in
simple_rl as it includes many interesting tricks.
"""

import sys

from simple_rl.agents import QLearningAgent, RandomAgent
from simple_rl.agents import Agent
from simple_rl.tasks import GridWorldMDP
from simple_rl.run_experiments import run_agents_on_mdp

import torch as th
from torch import nn
from torch import optim

import cherry as ch
from cherry.models.tabular import ActionValueFunction


class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden=128):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden, bias=False)
        self.linear2 = nn.Linear(hidden, output_size, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = th.tanh(x)
        x = self.linear2(x)
        x = th.tanh(x)
        return x


class CherryQAgent(Agent):

    def __init__(self, mdp, model, name='Cherry', lr=0.5):
        actions = mdp.get_actions()
        super(CherryQAgent, self).__init__(actions=actions, name=name)
        self.mdp = mdp
        self.action_size = len(actions)
        self.state_size = mdp.width * mdp.height
        self.anneal = 0.93
        self.model = model
        self.lr = lr
        self.reset()

    def _state(self, state):
        idx = (state.y - 1) * self.mdp.height
        idx += self.mdp.width
        return ch.onehot(idx, dim=self.state_size)

    def act(self, state, reward):
        state = self._state(state)
        reward = th.tensor([reward]).float()
        next_q = self.net(state).max()
        if self.curr_q is not None:
            # Compute loss and take optimization step
            residual = ch.td.temporal_difference(0.9,
                                                 rewards=reward,
                                                 dones=th.zeros(1),
                                                 values=self.curr_q,
                                                 next_values=next_q)
            loss = residual.pow(2).mul(0.5)
            loss.backward()
            self.opt.step()
            self.egreedy.epsilon *= self.anneal
            self.opt.zero_grad()

        # Now act
        q_values = self.net(state)
        action = self.egreedy(q_values).item()
        self.curr_q = q_values[0][action]
        return self.actions[action]

    def end_of_episode(self):
        self.curr_q = None

    def reset(self):
        super(CherryQAgent, self).reset()
        self.net = self.model(self.state_size, self.action_size)
        self.egreedy = ch.nn.EpsilonGreedy(epsilon=0.1)
        self.opt = optim.SGD(self.net.parameters(), lr=self.lr)
        self.curr_q = None


def main(open_plot=True):
    # Setup MDP.
    mdp = GridWorldMDP(width=4,
                       height=3,
                       init_loc=(1, 1),
                       goal_locs=[(4, 3)],
                       lava_locs=[(4, 2)],
                       gamma=0.95,
                       walls=[(2, 2)],
                       slip_prob=0.05)

    # Make agents.
    ql_agent = QLearningAgent(actions=mdp.get_actions())
    rand_agent = RandomAgent(actions=mdp.get_actions())
    tabular_agent = CherryQAgent(mdp,
                                 model=lambda *x: ActionValueFunction(*x, init=1.0),
                                 name='Tabular',
                                 lr=0.7)
    linear_agent = CherryQAgent(mdp,
                                model=lambda *x: nn.Linear(*x),
                                name='Linear',
                                lr=0.1)
    mlp_agent = CherryQAgent(mdp,
                             model=lambda *x: MLP(*x),
                             name='MLP',
                             lr=0.07)

    # Run experiment and make plot.
    agents = [rand_agent, ql_agent, tabular_agent, linear_agent, mlp_agent]
    run_agents_on_mdp(agents,
                      mdp,
                      instances=10,
                      episodes=50,
                      steps=50,
                      open_plot=open_plot)


if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
