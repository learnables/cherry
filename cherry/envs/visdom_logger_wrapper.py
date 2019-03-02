#!/usr/bin/env python3

import numpy as np
import cherry as ch

from .base import Wrapper

try:
    import visdom
except ImportError:
    visdom = ch.utils._ImportRaiser('Visdom', 'pip install visdom')


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
                 episode_interval=100,
                 name=None
                 ):
        """

        """
        super(VisdomLogger, self).__init__(env)
        self.interval = interval
        self.all_rewards = []
        self.all_dones = []
        self.ep_actions = []
        self.ep_states = []
        self.new_ep_actions = []
        self.new_ep_states = []
        self.num_steps = 0
        self.num_episodes = 0
        self.ep_length = 0
        self.values = {}
        self.values_plots = {}
        self.values_idx = {}

        # Instanciate visdom environment
        if name is None:
            name = env.spec.id
        self.visdom = visdom.Visdom(env=name)

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

        # Epsiode Actions
        opts = {
            'title': 'Actions on 1 Episode',
            'layoutopts': {
                'plotly': {
                    'xaxis': {'title': 'Episodes'},
                    'yaxis': {'title': 'Length'},
                    'zaxis': {'title': 'Magnitude'},
                }
            }
        }
        x = np.tile(np.arange(1, 101), (100, 1))
        y = x.transpose()
        X = np.exp((((x - 50) ** 2) + ((y - 50) ** 2)) / -(20.0 ** 2))
        self.ep_actions_plot = self.visdom.surf(X=X,
                                                opts=opts)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.all_rewards.append(reward)
        self.all_dones.append(done)
        self.new_ep_actions.append(action)
        self.new_ep_states.append(state)

        interval = self.interval > 0 and self.num_steps % self.interval == 0
        self._update(interval=interval)

        self.ep_length += 1
        self.num_steps += 1
        if done:
            self._reset_ep_stats()
        return state, reward, done, info

    def log(self, key, value, opts=None):
        if not key in self.values:
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

    def _update(self, interval=False):
        # Log immediate, non-self.values metrics
        if interval:
            y = np.array([self.all_rewards[-interval:]])
            x = np.array([self.num_steps])
            self.visdom.line(X=x,
                             Y=y,
                             win=self.rewards_plot,
                             update='append')
                            
