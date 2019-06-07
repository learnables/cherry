#!/usr/bin/env python3

import uuid
import numpy as np
import cherry as ch

from gym.spaces import Discrete

from .base import Wrapper
from .logger_wrapper import Logger

try:
    import visdom
except ImportError:
    visdom = ch._utils._ImportRaiser('Visdom', 'pip install visdom')


class VisdomLogger(Logger):

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
                 title=None
                 ):
        super(VisdomLogger, self).__init__(env=env,
                                           interval=interval,
                                           episode_interval=episode_interval,
                                           title=title)
        self.ep_actions = []
        self.full_ep_actions = []
        self.ep_renders = []
        self.full_ep_renders = []
        self.values_plots = {}
        self.discrete_actions = isinstance(env.action_space, Discrete)
        self.visdom = visdom.Visdom(env=self.title)
        self.ep_actions_win = str(uuid.uuid4())
        self.ep_renders_win = str(uuid.uuid4())

        self.can_record = 'rgb_array' in self.env.metadata['render.modes']

        # Mean rewards plot
        opts = {
            'title': 'Mean ' + str(self.ep_interval) + ' episode rewards',
            'layoutopts': {
                'plotly': {
                    'xaxis': {'title': 'Log'},
                    'yaxis': {'title': 'Rewards'},
                }
            }
        }
        self.values_plots['episode_rewards'] = self.visdom.line(X=np.empty(1),
                                                                Y=np.empty(1),
                                                                opts=opts)
        # Episode lengths plot
        opts = {
            'title': 'Mean ' + str(self.ep_interval) + ' episodes length',
            'fillarea': True,
            'layoutopts': {
                'plotly': {
                    'xaxis': {'title': 'Log'},
                    'yaxis': {'title': 'Length'},
                }
            }
        }
        self.values_plots['episode_lengths'] = self.visdom.line(X=np.empty(1),
                                                                Y=np.empty(1),
                                                                opts=opts)

    def update_ribbon_plot(self, ribbon_data, win_name):
        num_actions = self.action_size
        num_steps = len(ribbon_data)
        ribons = []
        x_t = []
        y_t = []
        z_t = []

        # get the data for each ribon (i.e. action)
        for i in range(num_actions):
            x_in = [None] * num_steps
            y_in = [None] * num_steps
            z_in = [None] * num_steps
            z_buff = float(ribbon_data[0][i])

            for j, step_action in enumerate(ribbon_data):
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
                           'win': win_name})

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action, *args, **kwargs):
        state, reward, done, info = super(VisdomLogger, self).step(action, *args, **kwargs)

        if self.interval > 0 and self.num_steps % self.interval == 0:
            self.update_steps_plots(info['logger_steps_stats'])
            self.update_ep_plots(info['logger_ep_stats'])

            if len(self.full_ep_actions) > 0:
                self.update_ribbon_plot(self.full_ep_actions,
                                        self.ep_actions_win)
            if len(self.full_ep_renders) > 0:
                try:
                    # TODO: Remove try clause when merged:
                    # https://github.com/facebookresearch/visdom/pull/595
                    frames = np.stack(self.full_ep_renders)
                    self.update_video(frames, self.ep_renders_win)
                    self.full_ep_renders = []
                except Exception:
                    pass

        # Should record ?
        if self.num_episodes % self.ep_interval == 0:
            if self.discrete_actions:
                action = ch.onehot(action, dim=self.action_size)[0]
            self.ep_actions.append(action)
            if self.can_record:
                frame = self.env.render(mode='rgb_array')
                self.ep_renders.append(frame)

        # Done recording ?
        if done and (self.num_episodes - 1) % self.ep_interval == 0:
            self.full_ep_actions = self.ep_actions
            self.ep_actions = []
            self.full_ep_renders = self.ep_renders
            self.ep_renders = []

        return state, reward, done, info

    def log(self, key, value, opts=None):
        super(VisdomLogger, self).log(key=key, value=value)
        # Create the plot
        if key not in self.values_plots:
            if opts is None:
                opts = {'title': key,
                        'layoutopts': {
                            'plotly': {
                                'xaxis': {'title': 'Log'},
                            }
                            }
                        }
            elif 'title' not in opts:
                opts['title'] = key
            self.values_plots[key] = self.visdom.line(X=np.empty(1),
                                                      Y=np.empty(1),
                                                      opts=opts)

    def update_steps_plots(self, stats):
        num_logs = len(self.all_rewards) // self.interval
        update = 'replace' if num_logs <= 1 else 'append'
        for key in stats:
            if key not in ['num_episodes', 'episode_lengths', 'episode_rewards']:
                x_values = np.zeros((1,)) + num_logs
                y_values = np.array([np.mean(stats[key])])
                self.visdom.line(X=x_values,
                                 Y=y_values,
                                 win=self.values_plots[key],
                                 update=update)

    def update_ep_plots(self, stats):
        num_logs = len(self.all_rewards) // self.interval
        update = 'replace' if num_logs <= 1 else 'append'
        for key in stats:
            if key is not 'num_episodes':
                x_values = np.zeros((1,)) + num_logs
                y_values = np.array([np.mean(stats[key])])
                self.visdom.line(X=x_values,
                                 Y=y_values,
                                 win=self.values_plots[key],
                                 update=update)

    def update_video(self, frames, win_name):
        self.visdom.video(frames, win=win_name,)
