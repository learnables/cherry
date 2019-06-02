#!/usr/bin/env python3

"""
**Description**

Helper functions for OpenAI Gym environments.
"""


import operator

from functools import reduce
from collections import OrderedDict

from gym.spaces import Box, Discrete, Dict


def get_space_dimension(space):
    """
    Returns the number of elements of a space sample, when unrolled.
    """
    msg = 'Space type not supported.'
    assert isinstance(space, (Box, Discrete, Dict)), msg
    if isinstance(space, Discrete):
        return space.n
    if isinstance(space, Box):
        return reduce(operator.mul, space.shape, 1)
    if isinstance(space, Dict):
        dimensions = {
            k[0]: get_space_dimension(k[1]) for k in space.spaces.items()
        }
        return OrderedDict(dimensions)
