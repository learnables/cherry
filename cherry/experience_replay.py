#!/usr/bin/env python3

import torch as th

from cherry.utils import totensor


class ExperienceReplay(object):

    def __init__(self,
                 states=None,
                 actions=None,
                 rewards=None,
                 next_states=None,
                 dones=None,
                 infos=None):
        self.storage = {
            'states': [] if states is None else states,
            'actions': [] if actions is None else actions,
            'rewards': [] if rewards is None else rewards,
            'next_states': [] if next_states is None else next_states,
            'dones': [] if dones is None else dones,
            'infos': [] if infos is None else infos,
        }

    def _access_property(self, name):
        values = self.storage[name]
        true_size = values[0].size()
        while true_size[0] == 1 and len(true_size) > 1:
            true_size = true_size[1:]
        return th.cat(values, dim=0).view(len(values), *true_size)

    def __len__(self):
        return len(self.storage['states'])

    def __getitem__(self, key):
        return {
            'state': self.storage['states'][key],
            'action': self.storage['actions'][key],
            'reward': self.storage['rewards'][key],
            'next_state': self.storage['next_states'][key],
            'done': self.storage['dones'][key],
            'info': self.storage['infos'][key],
        }

    @property
    def states(self):
        return self._access_property('states')

    @property
    def actions(self):
        return self._access_property('actions')

    @property
    def rewards(self):
        return self._access_property('rewards')

    @property
    def next_states(self):
        return self._access_property('next_states')

    @property
    def dones(self):
        return self._access_property('dones')

    @property
    def list_states(self):
        return self.storage['states']

    @property
    def list_actions(self):
        return self.storage['actions']

    @property
    def list_rewards(self):
        return self.storage['rewards']

    @property
    def list_next_states(self):
        return self.storage['next_states']

    @property
    def list_dones(self):
        return self.storage['dones']

    @property
    def list_infos(self):
        return self.storage['infos']

    def add(self, state, action, reward, next_state, done, info=None):
        self.storage['states'].append(totensor(state))
        self.storage['actions'].append(totensor(action))
        self.storage['rewards'].append(totensor(reward))
        self.storage['next_states'].append(totensor(next_state))
        self.storage['dones'].append(totensor(done))
        self.storage['infos'].append(info)

    def empty(self):
        self = self.__init__()
