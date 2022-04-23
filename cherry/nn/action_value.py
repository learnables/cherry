# -*- coding=utf-8 -*-

import torch


class ActionValue(torch.nn.Module):

    """
    <a href="https://github.com/learnables/cherry/blob/master/cherry/nn/action_value.py" class="source-link">[Source]</a>

    ## Description

    Abstract Module to represent Q-value functions.

    ## Example

    ~~~python
    class QValue(ActionValue):

        def __init__(self, state_size, action_size):
            super(QValue, self).__init__()
            self.mlp = MLP(state_size+action_size, 1, [1024, 1024])

        def forward(self, state, action):
            return self.mlp(torch.cat([state, action], dim=1))

    qf = QValue(128, 5)
    qvalue = qf(state, action)
    ~~~
    """

    def forward(self, state, action=None):
        """
        ## Description

        Returns the scalar value for taking action `action` in state `state`.

        If `action` is not given, should return the value for all actions (useful for DQN-like architectures).

        ## Arguments

        * `state` (Tensor) - State to be evaluated.
        * `action` (Tensor, *optional*, default=None) - Action to be evaluated.

        ## Returns

        * `value` (Tensor) - Value of taking `action` in `state`. Shape: (batch_size, 1)
        """
        raise NotImplementedError


class Twin(ActionValue):

    """
    <a href="https://github.com/learnables/cherry/blob/master/cherry/nn/action_value.py" class="source-link">[Source]</a>

    ## Description

    Helper class to implement Twin action-value functions as described in [1].

    ## References

    1. Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods". ICML 2018.

    ## Example

    ~~~python
    qvalue = Twin(QValue(), QValue())
    values = qvalue(states, actions)
    values1, values1 = qvalue.twin(states, actions)
    ~~~
    """

    def __init__(self, *action_values):
        """
        ## Arguments

        * `qvalue1, qvalue2, ...` (ActionValue) - Action value functions.
        """
        super(Twin, self).__init__()
        self.action_values = torch.nn.ModuleList(action_values)

    def twin(self, state, action):
        """
        ## Description

        Returns the values of each individual value function wrapped by this class.

        ## Arguments

        * `state` (Tensor) - State to be evaluated.
        * `action` (Tensor) - Action to be evaluated.
        """
        return [qf(state, action) for qf in self.action_values]

    def forward(self, state, action):
        """
        ## Description

        Returns the minimum value computed by the individual value functions wrapped by this class.

        ## Arguments

        * `state` (Tensor) - The state to evaluate.
        * `action` (Tensor) - The action to evaluate.
        """
        return torch.minimum(*self.twin(state, action))

    def all_action_values(self, state):

        return torch.minimum(*[
            qf.all_action_values(state) for qf in self.action_values
        ])
