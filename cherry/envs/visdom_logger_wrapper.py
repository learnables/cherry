#!/usr/bin/env python3

import numpy as np
import cherry as ch

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

        # Ribbon Plot
        # start with just a random plot
        opts =  {
                'title': 'Ribbon Plot',
                'layoutopts': {
                    'plotly': {
                        'xaxis': {'title': 'policy'},
                        'yaxis': {'title': 'time'},
                        'zaxis': {'title': 'activaion'},
                            }
                    }
                }

        #self.ep_ribbon_plot = self.visdom.line(X=np.zeros(1),
        #                                      Y=np.zeros(1),
        #                                      opts=opts)
        layout = dict(title="Ribbon Plot", xaxis={'title': 'x1'}, yaxis={'title': 'x2'}, zaxis={'title': 'z'}) 
        # create sample data points
        trace1 = dict(x=[[1, 2], [1,2], [1,2]], y=[[4, 4], [5, 5], [6, 6]], z=[[7,7], [8, 8], [9, 9]], type='surface', name='1st Trace')
        trace2 = dict(x=[[2, 3], [2,3], [2,3]], y=[[6, 6], [5, 5], [4, 4]], z=[[7,7], [8, 8], [9, 9]], type='surface', name='1st Trace')
        self.visdom._send({'data': [trace1, trace2], 'layout': layout, 'win': 'mywin'})

    # reads the list of actions that are indexed according to time step, then plot them and empty the action list
    def make_ribbon_plot(self):
        # check if it is continuous or discrete
        # get the number of ribbons (i.e. actions), and number of time steps
        discrete = False
        num_actions = None
        print("####### self.ep_actions[0] is of type: ",type(self.ep_actions[0]))
        if type(self.ep_actions[0]) != np.ndarray:
            discrete = True
            num_actions = int(np.max(self.ep_actions))
            print("----discrete")
        else:
            num_actions = len(self.ep_actions[0])
        num_steps = len(self.ep_actions)
       
        print("num actions: ", num_actions)
        print("num steps: ", num_steps)

        # create a dictionary for each ribon (serves as the trace for the action)
        ribons = []
       
        x_t = []
        y_t = []
        z_t = []

        # get the data for each ribon (i.e. action)
        for i in range(num_actions):
            x_in = [None] * num_steps
            y_in = [None] * num_steps
            z_in = [None] * num_steps

            z_buff = 0

            #print("for action ", i)
            for j, step_action in enumerate(self.ep_actions):
                x_in[j] = [i*2, i*2 + 1]
                y_in[j] = [j, j]
                if discrete:
                    if step_action == i:
                        z_buff = 1
                    else:
                        z_buff = 0
                else:
                    z_buff = 0.005 * float(step_action[i]) + 0.995 * z_buff

                z_in[j] = [z_buff, z_buff]
                print("----z_buff: ", z_buff)

            trace = dict(x=x_in, y=y_in, z=z_in, type='surface', name='')
            ribons.append(trace)
            print("x: ", x_in[:10], "\ny: ", y_in[:10], "\nz: ", z_in[:10])
            x_t = x_in
            y_t = y_in
            z_t = z_in

            #print("x: ", x_in[:10], "\ny: ", y_in[:10], "\nz: ", z_in[:10])
        
        # create the layout
        layout = dict(title="Ribbon Plot", xaxis={'title': 'x1'}, yaxis={'title': 'x2'}, zaxis={'title': 'z'}) 

        # send the trace, and layout to visdom 
        self.visdom._send({'data': ribons, 'layout': layout, 'win': 'mywin'})

        layout = dict(title="Ribbon Plot", xaxis={'title': 'x1'}, yaxis={'title': 'x2'}, zaxis={'title': 'z'}) 
        # create sample data points
        #import pdb;pdb.set_trace()
        #trace1 = dict(x=x_t, y=y_t, z=z_t, type='surface', name='1st Trace')
        #trace2 = dict(x=[[2, 3], [2,3], [2,3]], y=[[6, 6], [5, 5], [4, 4]], z=[[7,7], [8, 8], [9, 9]], type='surface', name='1st Trace')
        #self.visdom._send({'data': [trace1, trace2], 'layout': layout, 'win': 'mywin'})
        
        # reset the action list
        self.ep_actions = []

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.all_rewards.append(reward)
        self.all_dones.append(done)
        self.ep_actions.append(action)
        self.new_ep_actions.append(action)
        self.new_ep_states.append(state)

        #print("## the action taken was: ", action)

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

        # call the function to print the ribon plot
        self.make_ribbon_plot() 
    
    def _update(self, interval=False):
        # Log immediate, non-self.values metrics
        if interval:
            y = np.array([self.all_rewards[-interval:]])
            x = np.array([self.num_steps])
            self.visdom.line(X=x,
                             Y=y,
                             win=self.rewards_plot,
                             update='append')
