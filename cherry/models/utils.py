#!/usr/bin/env python3


def polyak_average(source, target, alpha):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/plot.py)

    **Description**

    Shifts the parameters of a model towards those of another one.

    Note: the parameter `alpha` indicates how much to shift in the direction of `target`.
    (i.e. the old parameters are kept at a rate of 1 - `alpha`.)
    
    **References**

    1. Polyak, B., and A. Juditsky. 1992. “Acceleration of Stochastic Approximation by Averaging.”

    **Arguments**

    * **source** (nn.Module) - The module to be shifted.
    * **target** (nn.Module) - The module indicating the shift direction.
    * **alpha** (float) - Strength of the shift.

    **Example**
    ~~~python
    target_qf = nn.Linear(23, 34)
    qf = nn.Linear(23, 34)
    ch.models.polyak_average(target_qf, qf, alpha=0.1)
    ~~~
    """
    for s, t in zip(source.parameters(), target.parameters()):
        s.data.mul_(1.0 - alpha).add_(alpha, t.data)
