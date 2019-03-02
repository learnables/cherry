#!/usr/bin/env python3

import operator
import numpy as np
import cherry as ch

from functools import reduce
from collections import OrderedDict

from gym.spaces import Box, Discrete, Dict


def flatten_state(space, state):
    """
    """
    if isinstance(space, Box):
        return np.asarray(state).flatten()
    if isinstance(space, Discrete):
        return ch.utils.onehot(state, space.n)
    raise('The space was not recognized.')


def get_space_dimension(space):
    """
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
