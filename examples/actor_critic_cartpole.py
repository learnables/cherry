#!/usr/bin/env python3

import torch
import cherry
import gym
import numpy as np
from itertools import count

SEED = 42

class A2C(torch.nn.Module):
    def __init__(self):
        super(A2C, self).__init__()
        
        self.gamma = 0.99
        self.vf_coef = 0.5
        self.ent_coef = 0.01

    def select_action(self, state):
        probs, value = self(state)
        mass = torch.distributions.Categorical(probs)
        action = mass.sample()
        # Return selected action, logprob, value estimation and categorical entropy
        return action, {"log_prob": mass.log_prob(action), "value": value, "entropy": mass.entropy()}

    
    def learn_step(self, replay, optimizer):
        policy_loss = []
        value_loss = []
        entropy_loss = []

        # Discount and normalize rewards
        rewards = cherry.td.discount(self.gamma, replay.reward(), replay.done())
        rewards = cherry.normalize(rewards)

        # Value function error (MSE)
        value_loss_fn = torch.nn.MSELoss()
        for sars, reward in zip(replay, rewards):
            log_prob = sars.log_prob
            value = sars.value
            entropy = sars.entropy

            # Compute advantage
            advantage = reward - value.squeeze(0)
            
            # Compute policy gradient loss
            # (advantage.detach() because you do not have to backward on the advantage path) 
            policy_loss.append(-log_prob * advantage.detach())
            # Compute value estimation loss
            value_loss.append(value_loss_fn(value.squeeze(0), reward))
            # Compute entropy loss
            entropy_loss.append(entropy)
        
        # Compute means over the accumulated errors
        policy_loss = torch.stack(policy_loss).mean()
        value_loss = torch.stack(value_loss).mean()
        entropy_loss = torch.stack(entropy_loss).mean()

        # Take an optimization step
        optimizer.zero_grad()
        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss
        loss.backward()
        optimizer.step()



class A2CPolicy(A2C):
    def __init__(self, state_size, action_size):
        super(A2CPolicy, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.n_hidden = 128

        # Backbone net
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.state_size, self.n_hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.n_hidden, self.n_hidden),
            torch.nn.LeakyReLU(),
        )

        # Action head (policy gradient)
        self.action_head = torch.nn.Sequential(
            torch.nn.Linear(self.n_hidden, self.action_size),
            torch.nn.Softmax(dim=1)
        )

        # Value estimation head (A2C)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(self.n_hidden, 1),
        )


    def forward(self, x):
        # Return both the action probabilities and the value estimations
        return self.action_head(self.net(x)), self.value_head(self.net(x))

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = cherry.envs.Logger(env, interval=1000)
    env = cherry.envs.Torch(env)
    env = cherry.envs.Runner(env)
    env.seed(SEED)

    policy = A2CPolicy(env.state_size, env.action_size)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    running_reward = 10
    for episode in count(1):
        replay = env.run(lambda state: policy.select_action(state), episodes=1)
        policy.learn_step(replay, optimizer)

        running_reward = running_reward * 0.99 + replay.reward().sum() * 0.01
        
        if running_reward > 190.0:
            print('Solved! Running reward now {} and '
                  'the last episode runs to {} time steps!'.format(running_reward,
                                                                   len(replay)))
            break
    
    while True:
        env.run(lambda state: policy.select_action(state), episodes=1, render=True)
