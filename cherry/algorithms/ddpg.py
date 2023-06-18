#!/usr/bin/env python3

import dataclasses
from torch.nn import functional as F
from cherry import debug

from .arguments import AlgorithmArguments


@dataclasses.dataclass
class DDPG(AlgorithmArguments):

    """
    <a href="https://github.com/learnables/cherry/blob/master/cherry/algorithms/ddpg.py" class="source-link">[Source]</a>

    ## Description

    Utilities to implement deep deterministic policy gradient algorithms from [1].

    ## References

    1. Lillicrap et al., "Continuous Control with Deep Reinforcement Learning", ICLR 2016.
    """

    @staticmethod
    def state_value_loss(values, next_values, rewards, dones, gamma):
        """
        ## Description

        The discounted Bellman loss, computed as:

        $$
        \\vert\\vert R + (1 - \\textrm{dones}) \\cdot \\gamma \\cdot V(s_{t+1}) - V(s_t) \\vert\\vert^2
        $$

        ## Arguments

        * `values` (tensor) - State values for timestep t.
        * `next_values` (tensor) - State values for timestep t+1.
        * `rewards` (tensor) - Vector of rewards for timestep t.
        * `dones` (tensor) - Termination flag.
        * `gamma` (float) - Discount factor.

        ## Returns

        * (tensor) - The state value loss above.


        """
        msg = 'rewards, values, and next_values must have equal size.'
        assert values.size() == next_values.size() == rewards.size(), msg
        if debug.IS_DEBUGGING:
            if rewards.requires_grad:
                debug.logger.warning('DDPG:state_value_loss: rewards.requires_grad is True.')
            if next_values.requires_grad:
                debug.logger.warning('DDPG:state_value_loss: next_values.requires_grad is True.')
            if not values.requires_grad:
                debug.logger.warning('DDPG:state_value_loss: values.requires_grad is False.')
        v_target = rewards + (1.0 - dones) * gamma * next_values
        return F.mse_loss(values, v_target)


state_value_loss = DDPG.state_value_loss
