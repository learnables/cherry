#!/usr/bin/env python3

from torch import nn


class RandomPolicy(nn.Module):

    """
    <a href="https://github.com/seba-1511/cherry/blob/master/cherry/models/utils.py" class="source-link">[Source]</a>

    ## Description

    Policy that randomly samples actions from the environment action space.

    ## Example

    ~~~python
    policy = ch.models.RandomPolicy(env)
    env = envs.Runner(env)
    replay = env.run(policy, steps=2048)
    ~~~
    """

    def __init__(self, env, *args, **kwargs):
        """
        ## Arguments

        * `env` (Environment) - Environment from which to sample actions.
        """
        super(RandomPolicy, self).__init__()
        self.env = env

    def forward(self, *args, **kwargs):
        return self.env.action_space.sample()


def polyak_average(source, target, alpha):
    """
    <a href="https://github.com/seba-1511/cherry/blob/master/cherry/models/utils.py" class="source-link">[Source]</a>

    ## Description

    Shifts the parameters of source towards those of target.

    Note: the parameter `alpha` indicates the convex combination weight of the source.
    (i.e. the old parameters are kept at a rate of `alpha`.)

    ## References

    1. Polyak, B., and A. Juditsky. 1992. “Acceleration of Stochastic Approximation by Averaging.”

    ## Arguments

    * `source` (Module) - The module to be shifted.
    * `target` (Module) - The module indicating the shift direction.
    * `alpha` (float) - Strength of the shift.

    ## Example

    ~~~python
    target_qf = nn.Linear(23, 34)
    qf = nn.Linear(23, 34)
    ch.models.polyak_average(target_qf, qf, alpha=0.9)
    ~~~
    """
    for s, t in zip(source.parameters(), target.parameters()):
        s.data.mul_(alpha).add_(t.data, alpha=1.0 - alpha)
