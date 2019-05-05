#!/usr/bin/env python3

import torch as th
import torch.nn as nn


class StateValueFunction(nn.Module):

    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/models/tabular.py)

    **Description**

    Stores a table of state values, V(s), one for each state.

    Assumes that the states are one-hot encoded.
    Also, the returned values are differentiable and can be used in
    conjunction with PyTorch's optimizers.

    **Arguments**

    * **state_size** (int) - The number of states in the environment.
    * **init** (function, *optional*, default=None) - The initialization
      scheme for the values in the table. (Default is 0.)

    **References**

    1. Sutton, Richard, and Andrew Barto. 2018. Reinforcement Learning, Second Edition. The MIT Press.

    **Example**
    ~~~python
    vf = StateValueFunction(env.state_size)
    state = env.reset()
    state = ch.onehot(state, env.state_size)
    state_value = vf(state)
    ~~~

    """

    def __init__(self, state_size, init=None):
        super(StateValueFunction, self).__init__()
        self.values = nn.Parameter(th.zeros((state_size, 1)))
        self.state_size = state_size
        if init is not None:
            if isinstance(init, (float, int, th.Tensor)):
                self.values.data.add_(init)
            else:
                init(self.values)

    def forward(self, state):
        return state.view(-1, self.state_size) @ self.values


class ActionValueFunction(nn.Module):

    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/models/tabular.py)

    **Description**

    Stores a table of action values, Q(s, a), one for each
    (state, action) pair.

    Assumes that the states and actions are one-hot encoded.
    Also, the returned values are differentiable and can be used in
    conjunction with PyTorch's optimizers.

    **Arguments**

    * **state_size** (int) - The number of states in the environment.
    * **action_size** (int) - The number of actions per state.
    * **init** (function, *optional*, default=None) - The initialization
      scheme for the values in the table. (Default is 0.)

    **References**

    1. Sutton, Richard, and Andrew Barto. 2018. Reinforcement Learning, Second Edition. The MIT Press.

    **Example**
    ~~~python
    qf = ActionValueFunction(env.state_size, env.action_size)
    state = env.reset()
    state = ch.onehot(state, env.state_size)
    all_action_values = qf(state)
    action = ch.onehot(0, env.action_size)
    action_value = qf(state, action)
    ~~~

    """

    def __init__(self, state_size, action_size, init=None):
        super(ActionValueFunction, self).__init__()
        self.values = nn.Parameter(th.zeros((state_size, action_size),
                                            requires_grad=True))
        self.state_size = state_size
        self.action_size = action_size
        if init is not None:
            if isinstance(init, (float, int, th.Tensor)):
                self.values.data.add_(init)
            else:
                init(self.values)

    def forward(self, state, action=None):
        action_values = (state @ self.values).view(-1, self.action_size)
        if action is None:
            return action_values
        return th.sum(action * action_values, dim=1, keepdim=True)
