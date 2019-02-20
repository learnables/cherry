#!/usr/bin/env python3

import random
import torch as th
from torch import Tensor as T

from cherry.utils import totensor, min_size


class Transition(dict):
    def __init__(self, **kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])
            self[k] = kwargs[k]
        self.__attributes = kwargs.keys()

    def __str__(self):
        name = 'Transition('
        for k in self.__attributes:
            name += k + '=' + str(getattr(self, k)) + ', '
        name += ')'
        return name

    def __repr__(self):
        return str(self)


class ExperienceReplay(list):

    def __init__(self,
                 states=None,
                 actions=None,
                 rewards=None,
                 next_states=None,
                 dones=None,
                 infos=None):
        list.__init__(self, [])
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
        true_size = min_size(values[0])
        return th.cat(values, dim=0).view(len(values), *true_size)

    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))

    def __len__(self):
        return len(self.storage['states'])

    def __str__(self):
        return 'ExperienceReplay(' + str(len(self)) + ')'

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        new_replay = ExperienceReplay()
        for sars in self:
            new_replay.add(**sars)
        for sars in other:
            new_replay.add(**sars)
        return new_replay

    def __iadd__(self, other):
        for sars in other:
            self.add(**sars)
        return self

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getattr__(self, attr):
        name = attr[:-1]
        values = [info[name] if name in info else None for info in self.infos]
        if isinstance(values[0], T):
            size = values[0].size()
            if all([isinstance(t, T) and t.size() == size for t in values]):
                true_size = min_size(values[0])
                return th.cat(values, dim=0).view(len(values), *true_size)
        return values

    def __getitem__(self, key):
        content = {
            'state': self.storage['states'][key],
            'action': self.storage['actions'][key],
            'reward': self.storage['rewards'][key],
            'next_state': self.storage['next_states'][key],
            'done': self.storage['dones'][key],
            'info': self.storage['infos'][key],
        }
        if isinstance(key, slice):
            return ExperienceReplay(
                states=content['state'][key],
                actions=content['action'][key],
                rewards=content['reward'][key],
                next_states=content['next_state'][key],
                dones=content['done'][key],
                infos=content['info'][key],
            )
        return Transition(**content)

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
    def infos(self):
        return self.storage['infos']

    def save(self, path):
        th.save(self.storage, path)

    def load(self, path):
        storage = th.load(path)
        self.storage = storage

    def add(self, state, action, reward, next_state, done, info=None):
        self.storage['states'].append(totensor(state))
        self.storage['actions'].append(totensor(action))
        self.storage['rewards'].append(totensor(reward))
        self.storage['next_states'].append(totensor(next_state))
        self.storage['dones'].append(totensor(done))
        self.storage['infos'].append(info)

    def update(self, fn):
        infos = self.storage['infos']
        attributes = ['state', 'action', 'reward', 'next_state', 'done']
        for i, sars in enumerate(self):
            update = fn(i, sars)
            for attr in update:
                if attr == 'info':
                    for info_name in update[attr]:
                        infos[i][info_name] = update['info'][info_name]
                elif attr in attributes:
                    self.storage[attr + 's'][i] = update[attr]
                else:
                    raise Exception('Update not properly formatted.')

    def sample(self, size=1, contiguous=False, episodes=False):
        """
        Samples from the Experience replay.

        Arguments:
            size: the number of samples.
            contiguous: whether to sample contiguous transitions.
            episodes: sample full episodes, instead of transitions.

        Return:
            ExperienceReplay()
        """
        if len(self) < 1 or size < 1:
            return ExperienceReplay()

        indices = []
        if episodes:
            if size > 1 and not contiguous:
                episodes = [self.sample(1, episodes=True) for _ in range(size)]
                content = {
                    'states': sum([e.storage['states'] for e in episodes], []),
                    'actions': sum([e.storage['actions'] for e in episodes], []),
                    'rewards': sum([e.storage['rewards'] for e in episodes], []),
                    'next_states': sum([e.storage['next_states'] for e in episodes], []),
                    'dones': sum([e.storage['dones'] for e in episodes], []),
                    'infos': sum([e.storage['infos'] for e in episodes], []),
                }
                return ExperienceReplay(**content)
            else:  # Sample 'size' contiguous episodes
                num_episodes = self.dones.sum().int().item()
                end = random.randint(size-1, num_episodes-size)
                # Find idx of the end-th done
                count = 0
                dones = self.dones
                for idx, d in reversed(list(enumerate(dones))):
                    if bool(d):
                        if count >= end:
                            end_idx = idx
                            break
                        count += 1
                # Fill indices
                indices.insert(0, end_idx)
                count = 0
                for idx in reversed(range(0, end_idx)):
                    if bool(dones[idx]):
                        count += 1
                        if count >= size:
                            break
                    indices.insert(0, idx)
        else:
            length = len(self) - 1
            if contiguous:
                start = random.randint(0, length - size)
                indices = list(range(start, start + size))
            else:
                indices = [random.randint(0, length) for _ in range(size)]

        # Fill the sample
        sample = ExperienceReplay()
        for idx in indices:
            transition = self[idx]
            sample.add(**transition)
        return sample

    def empty(self):
        self = self.__init__()
