#!/usr/bin/env python3

"""
Simple example of using cherry to solve cartpole.
The code is an adaptation of the PyTorch reinforcement learning example.
"""

import random
import gym
import numpy as np

from itertools import count

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

import inspect

import cherry as ch
import cherry.envs as envs
from cherry.policies import CategoricalPolicy
from cherry.rewards import discount_rewards
from cherry.utils import normalize

SEED = 567
GAMMA = 0.99
RENDER = False

random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


class Mlp(nn.Module):
    
    def identity(x):
        return x

    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            b_init_value=0.1,
    ):

        super(Mlp, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        hidden_init = self.fanin_init,
        self.fcs = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            fc.bias.data.fill_(b_init_value)
            self.fcs.append(fc)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)


    def forward(self, input):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        return output

    def fanin_init(self, tensor):
        size = tensor.size()
        if len(size) == 2:
            fan_in = size[0]
        elif len(size) > 2:
            fan_in = np.prod(size[1:])
        else:
            raise Exception("Shape must be have dimension at least 2.")
        bound = 1. / np.sqrt(fan_in)
        return tensor.data.uniform_(-bound, bound)


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """
    def forward(self, *args, **kwargs):
        flat_inputs = th.cat(args, dim=1)
        return super().forward(flat_inputs, **kwargs)


class SoftActorCritic():
    def __init__(
            self,
            env,
            policy,
            qf,
            vf,
            policy_optimizer,
            qf_optimizer,
            vf_optimizer,
            target_vf,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            discount=.99,

            train_policy_with_reparameterization=True,
            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            **kwargs
    ):

        self.policy = policy
        self.qf = qf
        self.vf = vf
        self.env = env
        self.train_policy_with_reparameterization = (
            train_policy_with_reparameterization
        )
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = th.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=policy_lr)

        self.target_vf = target_vf 
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = policy_optimizer
        self.qf_optimizer = qf_optimizer
        self.vf_optimizer = vf_optimizer
        
        self.discount = discount

    def update(self, replay):

        batch = replay.sample(40)
        log_pi = self.policy(batch.states).log_prob(batch.actions) 

        ''' Calculate Alpha Loss '''
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = 1
            alpha_loss = 0

        '''
        QF Loss

        Minimize Soft Bellman Residual, i.e.
        J_{Q}(theta) = E_{(s_t, a_t)~D}[grad[Q(a,s)](Q(s,a)-(r(s,a)+gamma(Q'(s', a')-alpha*log(pi(a'|s')))))
        and grad of J_{Q} is:
        grad(Q(t))*(Q(t) - (r(t) + gamma*(Q'(t+1) - alpha*log_pi(t+1))))

        '''

        qf_loss = 0
        for transition in batch:
            q_pred = self.qf(batch.states[transition].view(1,9), batch.actions[transition].view(1,9))
            target_v_value = self.target_vf(batch.next_states[transition].view(1,9))
            q_target = batch.rewards[transition] + (1 - batch.dones[transition]) * self.discount * target_v_value
            qf_loss += self.qf_criterion(q_pred, q_target)

        #   For some reason the loop is not running (/iterating of batch), causing qf_loss to
        #   always be zero 


        """
        VF Loss
        
        Below is code to calculate the VF Loss. Making it work is not a priority because of the note on
        page 6 of Haarnoja's paper from December 2018, indicating that the value function approximator is
        unnecessary.
        """

        '''q_new_actions = 0
        v_pred = []
        for transition in batch:
            q_new_actions += self.qf(batch.next_states[transition].view(1,9), batch.actions[transition].view(1,9))
            v_pred.append(th.tensor(self.vf(batch.states[transition])))

        v_pred = th.stack(v_pred)
            
        v_target = q_new_actions - alpha*log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())'''

        ''' Calculate Policy Loss '''

        policy_loss = 0
        for transition in batch:
            policy_loss += (alpha*log_pi[transition] - q_next_step)

        policy_loss /= batch.__len__()

        # TODO: calculate regression loss and add to policy_loss


        ''' Update Networks '''

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        '''self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()'''

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self._update_target_network()
    
    def np_to_pytorch_batch(self, np_batch):
        return {
            k: _elem_or_tuple_to_variable(x)
            for k, x in self._filter_batch(np_batch)
            if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
        }

    def _filter_batch(self, np_batch):
        for k, v in np_batch.items():
            if v.dtype == np.bool:
                yield k, v.astype(int)
            else:
                yield k, v

    def _elem_or_tuple_to_variable(self, elem_or_tuple):
        if isinstance(elem_or_tuple, tuple):
            return tuple(
                self._elem_or_tuple_to_variable(e) for e in elem_or_tuple
            )
        return self.from_numpy(elem_or_tuple).float()

    def from_numpy(*args, **kwargs):
        return th.from_numpy(*args, **kwargs).float().to(device)

#
# End SoftActorCritic Class
#


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = envs.Logger(env, interval=1000)
    env = envs.Torch(env)
    env.seed(SEED)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    net_size = 300

    policy = CategoricalPolicy(PolicyNet())
    qnet = FlattenMlp(hidden_sizes=[net_size, net_size], input_size=obs_dim+action_dim, output_size=1)
    vnet = FlattenMlp(hidden_sizes=[net_size, net_size], input_size=obs_dim, output_size=1)
    #target_vnet = FlattenMlp(hidden_sizes=[net_size, net_size], input_size=obs_dim, output_size=1)
    target_vnet = copy.deepcopy(vnet)

    policy_optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    qnet_optimizer = optim.Adam(qnet.parameters(), lr=1e-2)
    vnet_optimizer = optim.Adam(vnet.parameters(), lr=1e-2)

    critic = SoftActorCritic(env=env, policy=policy, qf=qnet, vf=vnet, policy_optimizer=policy_optimizer,
            qf_optimizer=qnet_optimizer, vf_optimizer=vnet_optimizer, target_vf=target_vnet)

    running_reward = 10.0
    replay = ch.ExperienceReplay()

    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            mass = policy(state)
            action = mass.sample()
            old_state = state
            state, reward, done, _ = env.step(action)
            replay.add(old_state, action, reward, state, done, info={
                'log_prob': mass.log_prob(action),  # Cache log_prob for later
            })
            if RENDER:
                env.render()
            if done:
                break

        # Compute termination criterion
        running_reward = running_reward * 0.99 + t * 0.01
        if running_reward > env.spec.reward_threshold:
            print('Solved! Running reward is now {} and '
                  'the last episode runs to {} time steps!'.format(running_reward, t))
            break

        # Update policy
        critic.update(replay)
        replay.empty()

#####################################################################################################
#   SAC To Dos
#####################################################################################################
# Use to target Q functions to eliminate positive bias (SACAA 6)

