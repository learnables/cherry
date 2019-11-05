#!/usr/bin/env python3

import torch as th
import torch.nn as nn

from cherry.nn import RoboticsLinear


class RoboticsMLP(nn.Module):

    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/models/robotics.py)

    **Description**

    A multi-layer perceptron with proper initialization for robotic control.

    **Credit**

    Adapted from Ilya Kostrikov's implementation.

    **Arguments**

    * **inputs_size** (int) - Size of input.
    * **output_size** (int) - Size of output.
    * **layer_sizes** (list, *optional*, default=None) - A list of ints,
      each indicating the size of a hidden layer.
      (Defaults to two hidden layers of 64 units.)

    **Example**
    ~~~python
    target_qf = ch.models.robotics.RoboticsMLP(23,
                                             34,
                                             layer_sizes=[32, 32])
    ~~~
    """

    def __init__(self, input_size, output_size, layer_sizes=None):
        super(RoboticsMLP, self).__init__()
        if layer_sizes is None:
            layer_sizes = [64, 64]
        if len(layer_sizes) > 0:
            layers = [RoboticsLinear(input_size, layer_sizes[0]),
                      nn.Tanh()]
            for in_, out_ in zip(layer_sizes[:-1], layer_sizes[1:]):
                layers.append(RoboticsLinear(in_, out_))
                layers.append(nn.Tanh())
            layers.append(RoboticsLinear(layer_sizes[-1], output_size))
        else:
            layers = [RoboticsLinear(input_size, output_size)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class RoboticsActor(RoboticsMLP):

    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/models/robotics.py)

    **Description**

    A multi-layer perceptron with initialization designed for choosing
    actions in continuous robotic environments.

    **Credit**

    Adapted from Ilya Kostrikov's implementation.

    **Arguments**

    * **inputs_size** (int) - Size of input.
    * **output_size** (int) - Size of action size.
    * **layer_sizes** (list, *optional*, default=None) - A list of ints,
      each indicating the size of a hidden layer.
      (Defaults to two hidden layers of 64 units.)

    **Example**
    ~~~python
    policy_mean = ch.models.robotics.Actor(28,
                                          8,
                                          layer_sizes=[64, 32, 16])
    ~~~
    """

    def __init__(self, input_size, output_size, layer_sizes=None):
        super(RoboticsMLP, self).__init__()
        if layer_sizes is None:
            layer_sizes = [64, 64]
        if len(layer_sizes) > 0:
            layers = [RoboticsLinear(input_size, layer_sizes[0]),
                      nn.Tanh()]
            for in_, out_ in zip(layer_sizes[:-1], layer_sizes[1:]):
                layers.append(RoboticsLinear(in_, out_))
                layers.append(nn.Tanh())
            layers.append(RoboticsLinear(layer_sizes[-1],
                                         output_size,
                                         gain=1.0))
        else:
            layers = [RoboticsLinear(input_size, output_size, gain=1.0)]
        self.layers = nn.Sequential(*layers)


class LinearValue(nn.Module):

    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/models/robotics.py)

    **Description**

    A linear state-value function, whose parameters are found by minimizing
    least-squares.

    **Credit**

    Adapted from Tristan Deleu's implementation.

    **References**

    1. Duan et al. 2016. “Benchmarking Deep Reinforcement Learning for Continuous Control.”
    2. [https://github.com/tristandeleu/pytorch-maml-rl](https://github.com/tristandeleu/pytorch-maml-rl)

    **Arguments**

    * **inputs_size** (int) - Size of input.
    * **reg** (float, *optional*, default=1e-5) - Regularization coefficient.

    **Example**
    ~~~python
    states = replay.state()
    rewards = replay.reward()
    dones = replay.done()
    returns = ch.td.discount(gamma, rewards, dones)
    baseline = LinearValue(input_size)
    baseline.fit(states, returns)
    next_values = baseline(replay.next_states())
    ~~~
    """

    def __init__(self, input_size, reg=1e-5):
        super(LinearValue, self).__init__()
        self.linear = nn.Linear(2 * input_size + 4, 1, bias=False)
        self.reg = reg

    def _features(self, states):
        length = states.size(0)
        ones = th.ones(length, 1).to(states.device)
        al = th.arange(length, dtype=th.float32, device=states.device).view(-1, 1) / 100.0
        return th.cat([states, states**2, al, al**2, al**3, ones], dim=1)

    def fit(self, states, returns):
        features = self._features(states)
        reg = self.reg * th.eye(features.size(1))
        reg = reg.to(states.device)
        A = features.t() @ features + reg
        b = features.t() @ returns
        if hasattr(th, 'lstsq'):  # Required for torch < 1.3.0
            coeffs, _ = th.lstsq(b, A)
        else:
            coeffs, _ = th.gels(b, A)
        self.linear.weight.data = coeffs.data.t()

    def forward(self, states):
        features = self._features(states)
        return self.linear(features)
