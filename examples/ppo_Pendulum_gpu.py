import gym
import torch
import random
import numpy as np
from torch import optim
from torch import nn
from torch.distributions import Normal

import cherry as ch
from cherry import td
from cherry import pg
from cherry import envs
import time

DISCOUNT = 0.99
EPSILON = 0.05
HIDDEN_SIZE = 32
LEARNING_RATE = 0.01
MAX_STEPS = 7000
BATCH_SIZE = 2048
TRACE_DECAY = 0.97
SEED = 42
PPO_CLIP_RATIO = 0.2
PPO_EPOCHS = 20
REPLAY_SIZE = 100000
USE_CUDA = False

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if USE_CUDA and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)


class Actor(nn.Module):
    def __init__(self, hidden_size, stochastic=True, layer_norm=False):
        super().__init__()
        layers = [nn.Linear(3, hidden_size),
                  nn.Tanh(),
                  nn.Linear(hidden_size, hidden_size),
                  nn.Tanh(),
                  nn.Linear(hidden_size, 1)]
        if layer_norm:
            layers = (layers[:1]
                      + [nn.LayerNorm(hidden_size)]
                      + layers[1:3]
                      + [nn.LayerNorm(hidden_size)]
                      + layers[3:])
        self.policy = nn.Sequential(*layers)
        if stochastic:
            self.policy_log_std = nn.Parameter(torch.tensor([[0.]]))

    def forward(self, state):
        policy = self.policy(state)
        return policy


class Critic(nn.Module):
    def __init__(self, hidden_size, state_action=False, layer_norm=False):
        super().__init__()
        self.state_action = state_action
        layers = [nn.Linear(3 + (1 if state_action else 0), hidden_size),
                  nn.Tanh(),
                  nn.Linear(hidden_size, hidden_size),
                  nn.Tanh(),
                  nn.Linear(hidden_size, 1)]
        if layer_norm:
            layers = (layers[:1]
                      + [nn.LayerNorm(hidden_size)]
                      + layers[1:3]
                      + [nn.LayerNorm(hidden_size)]
                      + layers[3:])
        self.value = nn.Sequential(*layers)

    def forward(self, state, action=None):
        if self.state_action:
            value = self.value(torch.cat([state, action], dim=1))
        else:
            value = self.value(state)
        return value.squeeze(dim=1)


class ActorCritic(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.actor = Actor(hidden_size, stochastic=True)
        self.critic = Critic(hidden_size)

    def forward(self, state):
        policy = Normal(self.actor(state), self.actor.policy_log_std.exp())
        value = self.critic(state)
        action = policy.sample()
        log_prob = policy.log_prob(action)
        return action, {
                'mass': policy,
                'log_prob': log_prob,
                'value': value,
        }


def main(env='Pendulum-v0'):
    agent = ActorCritic(HIDDEN_SIZE).to(device)
    agent.apply(weights_init)
    
    actor_optimizer = optim.Adam(agent.actor.parameters(), lr=LEARNING_RATE)
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=LEARNING_RATE)
    actor_scheduler = torch.optim.lr_scheduler.StepLR(actor_optimizer, step_size=2000, gamma=0.5)
    critic_scheduler = torch.optim.lr_scheduler.StepLR(critic_optimizer, step_size=2000, gamma=0.5)
    replay = ch.ExperienceReplay()

    env = gym.make(env)
    env.seed(SEED)
    env = envs.Torch(env)
    env = envs.Logger(env)
    env = envs.Runner(env)
    replay = ch.ExperienceReplay()
    
    def get_action(state):
        return agent(state.to(device))

    for step in range(1, MAX_STEPS + 1):
        replay += env.run(get_action, episodes=1)

        if len(replay) >= BATCH_SIZE:
            #batch = replay.sample(BATCH_SIZE).to(device)
            batch = replay.to(device)
            with torch.no_grad():
                advantages = pg.generalized_advantage(DISCOUNT,
                                                      TRACE_DECAY,
                                                      batch.reward(),
                                                      batch.done(),
                                                      batch.value(),
                                                      torch.zeros(1).to(device))
                advantages = ch.normalize(advantages, epsilon=1e-8)
                returns = td.discount(DISCOUNT,
                                      batch.reward(),
                                      batch.done())
                old_log_probs = batch.log_prob()

            new_values = batch.value()
            new_log_probs = batch.log_prob()
            for epoch in range(PPO_EPOCHS):
                # Recalculate outputs for subsequent iterations
                if epoch > 0:
                    _, infos = agent(batch.state())
                    masses = infos['mass']
                    new_values = infos['value'].view(-1, 1)
                    new_log_probs = masses.log_prob(batch.action())

                # Update the policy by maximising the PPO-Clip objective
                policy_loss = ch.algorithms.ppo.policy_loss(new_log_probs,
                                                            old_log_probs,
                                                            advantages,
                                                            clip=PPO_CLIP_RATIO)
                actor_optimizer.zero_grad()
                policy_loss.backward()
                #nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
                actor_optimizer.step()

                # Fit value function by regression on mean-squared error
                value_loss = ch.algorithms.a2c.state_value_loss(new_values,
                                                                returns)
                critic_optimizer.zero_grad()
                value_loss.backward()
                #nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
                critic_optimizer.step()
            
            actor_scheduler.step()
            critic_scheduler.step()

            replay.empty()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("device:{}, time elapsed:{}".format(device, time.time() - start_time))

