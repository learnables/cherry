#!/usr/bin/env python3

import uuid
import numpy as np
import cherry as ch

from gym.spaces import Discrete

from .base import Wrapper

try:
    import visdom
except ImportError:
    visdom = ch._utils._ImportRaiser('Visdom', 'pip install visdom')


class VisdomLogger(Wrapper):

    """
    Enables logging and debug values to Visdom.

    Arguments

    * env: The environment to wrap.
    * interval: (int) Update frequency for episodes.


    """

    def __init__(self,
                 env,
                 interval=1000,
                 episode_interval=10,
                 name=None
                 ):
        super(VisdomLogger, self).__init__(env)
        self.interval = interval
        self.episode_interval = episode_interval
        self.all_rewards = []
        self.all_dones = []
        self.ep_actions = []
        self.ep_states = []
        self.num_steps = 0
        self.num_episodes = 0
        self.ep_length = 0
        self.values = {}
        self.values_plots = {}
        self.values_idx = {}
        self.discrete_actions = isinstance(env.action_space, Discrete)
        self.ep_actions_win = str(uuid.uuid4())

        # Instanciate visdom environment
        if name is None:
            name = env.spec.id
        self.visdom = visdom.Visdom(env=name)
        self.name = name

        # Mean rewards plot
        opts = {
            'title': 'Mean Last ' + str(interval) + ' Rewards',
            'layoutopts': {
                'plotly': {
                    'xaxis': {'title': 'Steps'},
                    'yaxis': {'title': 'Rewards'},
                }
            }
        }
        self.rewards_plot = self.visdom.line(X=np.zeros(1),
                                             Y=np.zeros(1),
                                             opts=opts)
        # Episode lengths plot
        opts = {
            'title': 'Episodes Length',
            'fillarea': True,
            'layoutopts': {
                'plotly': {
                    'xaxis': {'title': 'Episodes'},
                    'yaxis': {'title': 'Length'},
                }
            }
        }
        self.ep_length_plot = self.visdom.line(X=np.zeros(1),
                                               Y=np.zeros(1),
                                               opts=opts)

    def plot_ep_actions(self):
        discrete = isinstance(self.action_space, Discrete)
        num_actions = self.action_size
        num_steps = len(self.ep_actions)
        ribons = []
        x_t = []
        y_t = []
        z_t = []

        # get the data for each ribon (i.e. action)
        for i in range(num_actions):
            x_in = [None] * num_steps
            y_in = [None] * num_steps
            z_in = [None] * num_steps
            z_buff = float(self.ep_actions[0][i])

            for j, step_action in enumerate(self.ep_actions):
                x_in[j] = [i*2, i*2 + 1]
                y_in[j] = [j, j]
                z_buff = 0.5 * z_buff + 0.5 * float(step_action[i])
                z_in[j] = [z_buff, z_buff]

            trace = dict(x=x_in,
                         y=y_in,
                         z=z_in,
                         type='surface',
                         name='')
            ribons.append(trace)
            x_t = x_in
            y_t = y_in
            z_t = z_in

        layout = dict(title='Actions over 1 Episode',
                      xaxis={'title': 'Policy'},
                      yaxis={'title': 'Time'},
                      zaxis={'title': 'Activation'})

        # send the trace, and layout to visdom
        self.visdom._send({'data': ribons,
                           'layout': layout,
                           'win': self.ep_actions_win})

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.all_rewards.append(reward)
        self.all_dones.append(done)
        if self.discrete_actions:
            action = ch.onehot(action, dim=self.action_size)[0]
        self.ep_actions.append(action)

        interval = self.interval > 0 and self.num_steps % self.interval == 0
        self._update(interval=interval)

        self.ep_length += 1
        self.num_steps += 1
        if done:
            self._reset_ep_stats()
        return state, reward, done, info

    def log(self, key, value, opts=None):
        if key not in self.values:
            if opts is None:
                opts = {'title': key}
            elif 'title' not in opts:
                opts['title'] = key
            self.values[key] = []
            self.values_plots[key] = self.visdom.line(X=np.zeros(1),
                                                      Y=np.zeros(1),
                                                      opts=opts)
            self.values_idx[key] = 0
            setattr(self, key, self.values[key])
        self.values[key].append(value)
        new_data = self.values[key][self.values_idx[key]:]
        x_values = self.values_idx[key] + np.arange(0, len(new_data))
        self.visdom.line(X=x_values,
                         Y=np.array(new_data),
                         win=self.values_plots[key],
                         update='append')
        self.values_idx[key] = len(self.values[key])

    def _reset_ep_stats(self):
        self.num_episodes += 1
        self.visdom.line(X=np.array([self.num_episodes]),
                         Y=np.array([self.ep_length]),
                         win=self.ep_length_plot,
                         update='append')
        self.ep_length = 0
        if self.num_episodes % self.episode_interval == 0:
            self.plot_ep_actions()
        self.ep_actions = []

    def _update(self, interval=False):
        # Log immediate, non-self.values metrics
        if interval:
            y = np.array([self.all_rewards[-interval:]])
            x = np.array([self.num_steps])
            self.visdom.line(X=x,
                             Y=y,
                             win=self.rewards_plot,
                             update='append')
