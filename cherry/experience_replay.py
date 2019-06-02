#!/usr/bin/env python3

import random
import torch as th

import cherry as ch
from cherry._utils import _istensorable, _min_size

"""
TODO: replay.astype(dtype) + init dtype
"""


class Transition(object):

    """

    **Description**

    Represents a (s, a, r, s', d) tuple.

    All attributes (including the ones in infos) are accessible via
    `transition.name_of_attr`.
    (e.g. `transition.log_prob` if `log_prob` is in `infos`.)

    **Arguments**

    * **state** (tensor) - Originating state.
    * **action** (tensor) - Executed action.
    * **reward** (tensor) - Observed reward.
    * **next_state** (tensor) - Resulting state.
    * **done** (tensor) - Is `next_state` a terminal (absorbing) state ?
    * **infos** (dict, *optional*, default=None) - Additional information on
      the transition.

    **Example**

    ~~~python
    for transition in replay:
        print(transition.state)
    ~~~
    """

    def __init__(self,
                 state,
                 action,
                 reward,
                 next_state,
                 done,
                 device=None,
                 **infos):
        self.__fields = ['state', 'action', 'reward', 'next_state', 'done']
        values = [state, action, reward, next_state, done]
        for key, val in zip(self.__fields, values):
            setattr(self, key, val)
        info_keys = infos.keys()
        self.__fields += info_keys
        for key in info_keys:
            setattr(self, key, infos[key])
        self.device = device

    def __str__(self):
        string = 'Transition(' + ', '.join(self.__fields)
        if self.device is not None:
            string += ', device=\'' + str(self.device) + '\''
        string += ')'
        return string

    def __repr__(self):
        return str(self)

    def cpu(self):
        return self.to('cpu')

    def cuda(self, device=0, *args, **kwargs):
        return self.to('cuda:' + str(device), *args, **kwargs)

    def _apply(self, fn, device=None):
        if device is None:
            device = self.device
        new_transition = {'device': device}
        for field in self.__fields:
            value = getattr(self, field)
            if isinstance(value, th.Tensor):
                new_transition[field] = fn(value)
            else:
                new_transition[field] = value
        return Transition(**new_transition)

    def to(self, *args, **kwargs):
        """
        **Description**

        Moves the constituents of the transition to the desired device,
        and casts them to the desired format.

        Note: This is done in-place and doesn't create a new transition.

        **Arguments**

        * **device** (device, *optional*, default=None) - The device to move the data to.
        * **dtype** (dtype, *optional*, default=None) - The torch.dtype format to cast to.
        * **non_blocking** (bool, *optional*, default=False) - Whether to perform the move asynchronously.

        **Example**

        ~~~python
        sars = Transition(state, action, reward, next_state)
        sars.to('cuda')
        ~~~

        """
        device, dtype, non_blocking = th._C._nn._parse_to(*args, **kwargs)
        return self._apply(lambda t: t.to(device, dtype if t.is_floating_point() else None, non_blocking), device)

    def half(self):
        return self._apply(lambda t: t.half() if t.is_floating_point() else t)

    def double(self):
        return self._apply(lambda t: t.double() if t.is_floating_point() else t)


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
                  density: action_density,
                  log_prob: action_density.log_prob(action),
                  )

    replay.state()  # Tensor of states
    replay.action()  # Tensor of actions
    replay.density()  # list of action_density
    replay.log_prob()  # Tensor of log_probabilities

    new_replay = replay[-10:]  # Last 10 transitions in new_replay

    #Sample some previous experience
    batch = replay.sample(32, contiguous=True)
    ~~~
    """

    def __init__(self, storage=None, device=None):
        list.__init__(self)
        if storage is None:
            storage = []
        self._storage = storage
        self.device = device

    def _access_property(self, name):
        try:
            values = [getattr(sars, name) for sars in self._storage]
        except AttributeError:
            msg = 'Attribute ' + name + ' not in replay.'
            raise AttributeError(msg)
        true_size = _min_size(values[0])
        return th.cat(values, dim=0).view(len(values), *true_size)

    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))

    def __len__(self):
        return len(self._storage)

    def __str__(self):
        string = 'ExperienceReplay(' + str(len(self))
        if self.device is not None:
            string += ', device=\'' + str(self.device) + '\''
        string += ')'
        return string

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        storage = self._storage + other._storage
        return ExperienceReplay(storage)

    def __iadd__(self, other):
        self._storage += other._storage
        return self

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getattr__(self, attr):
        return lambda: self._access_property(attr)

    def __getitem__(self, key):
        value = self._storage[key]
        if isinstance(key, slice):
            return ExperienceReplay(value)
        return value

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
        th.save(self._storage, path)

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
        self._storage = th.load(path)

    def append(self,
               state=None,
               action=None,
               reward=None,
               next_state=None,
               done=None,
               **infos):
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
        * **infos** (dict, *optional*, default=None) - Additional information
          on the transition.

        **Example**
        ~~~python
        replay.append(state, action, reward, next_state, done, info={
            'density': density,
            'log_prob': density.log_prob(action),
        })
        ~~~
        """
        for key in infos:
            if _istensorable(infos[key]):
                infos[key] = ch.totensor(infos[key])
        sars = Transition(ch.totensor(state),
                          ch.totensor(action),
                          ch.totensor(reward),
                          ch.totensor(next_state),
                          ch.totensor(done),
                          **infos)
        self._storage.append(sars.to(self.device))

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
                replay = ExperienceReplay()
                return sum([self.sample(1, episodes=True) for _ in range(size)], replay)
            else:  # Sample 'size' contiguous episodes
                num_episodes = self.done().sum().int().item()
                end = random.randint(size-1, num_episodes-size)
                # Find idx of the end-th done
                count = 0
                dones = self.done()
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
        storage = [self[idx] for idx in indices]
        return ExperienceReplay(storage)

    def empty(self):
        """
        **Description**

        Removes all data from an ExperienceReplay.

        **Example**
        ~~~python
        replay.empty()
        ~~~
        """
        self._storage = []

    def cpu(self):
        return self.to('cpu')

    def cuda(self, device=0, *args, **kwargs):
        return self.to('cuda:' + str(device), *args, **kwargs)

    def to(self, *args, **kwargs):
        """
        **Description**

        Calls `.to()` on all transitions of the experience replay, moving them to the
        desired device and casting the to the desired format.

        Note: This return a new experience replay, but the transitions are modified in-place.

        **Arguments**

        * **device** (device, *optional*, default=None) - The device to move the data to.
        * **dtype** (dtype, *optional*, default=None) - The torch.dtype format to cast to.
        * **non_blocking** (bool, *optional*, default=False) - Whether to perform the move asynchronously.

        **Example**

        ~~~python
        replay.to('cuda:1')
        policy.to('cuda:1')
        for sars in replay:
            cuda_action = policy(sars.state).sample()
        ~~~

        """
        device, dtype, non_blocking = th._C._nn._parse_to(*args, **kwargs)
        storage = [sars.to(*args, **kwargs) for sars in self._storage]
        return ExperienceReplay(storage, device=device)

    def half(self):
        storage = [sars.half() for sars in self._storage]
        return ExperienceReplay(storage, device=self.device)

    def double(self):
        storage = [sars.double() for sars in self._storage]
        return ExperienceReplay(storage, device=self.device)
