#!/usr/bin/env python3

import random
import torch as th
from torch import Tensor as T

from cherry.utils import totensor, min_size


class Transition(dict):

    """
    **Description**

    Represents a (s, a, r, s', d) tuple.

    **Arguments**

    * **state** (tensor) - Originating state.
    * **action** (tensor) - Executed action.
    * **reward** (tensor) - Observed reward.
    * **next_state** (tensor) - Resulting state.
    * **done** (tensor) - Is `next_state` a terminal (absorbing) state ?
    * **info** (dict, *optional*, default=None) - Additional information on
      the transition.

    **Example**

    ~~~python
    for transition in replay:
        print(transition.state)
    ~~~
    """

    def __init__(self, state, action, reward, next_state, done, info=None):
        self.__attributes = ['state',
                             'action',
                             'reward',
                             'next_state',
                             'done',
                             'info']
        self['state'] = state
        self['action'] = action
        self['reward'] = reward
        self['next_state'] = next_state
        self['done'] = done
        self['info'] = info
        for attr in self.__attributes:
            setattr(self, attr, self[attr])

    def __str__(self):
        name = 'Transition('
        for k in self.__attributes:
            name += k + '=' + str(getattr(self, k)) + ', '
        name += ')'
        return name

    def __repr__(self):
        return str(self)


class ExperienceReplay(list):

    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/experience_replay.py)

    **Description**

    Experience replay buffer to store, retrieve, and sample past transitions.

    `ExperienceReplay` behaves like a list of transitions, .
    It also support accessing specific properties, such as states, actions,
    rewards, next_states, and informations.
    The first four are returned as tensors, while `infos` is returned as
    a list of dicts.
    The properties of infos can be accessed directly by appending an `s` to
    their dictionary key -- see Examples below.
    In this case, if the values of the infos are tensors, they will be returned
    as a concatenated Tensor.
    Otherwise, they default to a list of values.

    **Arguments**

    * **states** (Tensor, *optional*, default=None) - Tensor of states.
    * **actions** (Tensor, *optional*, default=None) - Tensor of actions.
    * **rewards** (Tensor, *optional*, default=None) - Tensor of rewards.
    * **next_states** (Tensor, *optional*, default=None) - Tensor of
      next_states.
    * **dones** (Tensor, *optional*, default=None) - Tensor of dones.
    * **infos** (list, *optional*, default=None) - List of infos.

    **References**

    1. Lin, Long-Ji. 1992. “Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and Teaching.” Machine Learning 8 (3): 293–321.

    **Example**

    ~~~python
    replay = ch.ExperienceReplay()  # Instanciate a new replay
    replay.append(state,  # Add experience to the replay
                  action,
                  reward,
                  next_state,
                  done,
                  info={
                       'density': action_density,
                       'log_prob': action_density.log_prob(action),
                  })

    replay.state  # Tensor of states
    replay.actions  # Tensor of actions
    replay.densitys  # list of action_density
    replay.log_probs  # Tensor of log_probabilities

    new_replay = replay[-10:]  # Last 10 transitions in new_replay

    #Sample some previous experience
    batch = replay.sample(32, contiguous=True)
    ~~~
    """

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
            new_replay.append(**sars)
        for sars in other:
            new_replay.append(**sars)
        return new_replay

    def __iadd__(self, other):
        for sars in other:
            self.append(**sars)
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
                return th.stack(values, dim=0).view(len(values), *true_size)
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
                states=content['state'],
                actions=content['action'],
                rewards=content['reward'],
                next_states=content['next_state'],
                dones=content['done'],
                infos=content['info'],
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
        """
        **Description**

        Serializes and saves the ExperienceReplay into the given path.

        **Arguments**

        * **path** (str) - File path.

        **Example**
        ~~~python
        replay.save('my_replay_file.pt')
        ~~~
        """
        th.save(self.storage, path)

    def load(self, path):
        """
        **Description**

        Loads data from a serialized ExperienceReplay.

        **Arguments**

        * **path** (str) - File path of serialized ExperienceReplay.

        **Example**
        ~~~python
        replay.load('my_replay_file.pt')
        ~~~
        """
        storage = th.load(path)
        self.storage = storage

    def append(self, state, action, reward, next_state, done, info=None):
        """
        **Description**

        Appends new data to the list ExperienceReplay.

        **Arguments**

        * **state** (tensor/ndarray/list) - Originating state.
        * **action** (tensor/ndarray/list) - Executed action.
        * **reward** (tensor/ndarray/list) - Observed reward.
        * **next_state** (tensor/ndarray/list) - Resulting state.
        * **done** (tensor/bool) - Is `next_state` a terminal (absorbing)
          state ?
        * **info** (dict, *optional*, default=None) - Additional information on
          the transition.

        **Example**
        ~~~python
        replay.append(state, action, reward, next_state, done, info={
            'density': density,
            'log_prob': density.log_prob(action),
        })
        ~~~
        """
        self.storage['states'].append(totensor(state))
        self.storage['actions'].append(totensor(action))
        self.storage['rewards'].append(totensor(reward))
        self.storage['next_states'].append(totensor(next_state))
        self.storage['dones'].append(totensor(done))
        self.storage['infos'].append(info)

    def add(self, *args, **kwargs):
        """
        **Description**

        (Deprecated) Alias for .append()
        """
        self.append(*args, **kwargs)

    def update(self, fn):
        """
        **Description**

        Updates all samples in the replay according to `fn`.

        `fn` should take two arguments and returns a dict of updated values.
        The first one corresponds to the index of the transition to be updated.
        The second is the transition itself.

        Note: You should return the updated values, not modify the values
              in-place on the transition.

        **Arguments**

        * **fn** (function) - Update function.

        **Example**
        ~~~python
        replay.update(lambda i, sars: {
            'reward': rewards[i].detach(),
            'info': {
                'advantage': advantages[i].detach()
            },
        })
        ~~~
        """
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

        **Arguments**

        * **size** (int, *optional*, default=1) - The number of samples.
        * **contiguous** (bool, *optional*, default=False) - Whether to sample
          contiguous transitions.
        * **episodes** (bool, *optional*, default=False) - Sample full
          episodes, instead of transitions.

        **Return**

        * ExperienceReplay - New ExperienceReplay containing the sampled
          transitions.
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
            sample.append(**transition)
        return sample

    def empty(self):
        """
        **Description**

        Removes all data from an ExperienceReplay.

        **Example**
        ~~~python
        replay.empty()
        ~~~
        """
        self = self.__init__()
