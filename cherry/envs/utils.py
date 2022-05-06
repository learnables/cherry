#!/usr/bin/env python3

"""
**Description**

Helper functions for OpenAI Gym environments.
"""


import operator
import numpy as np

from functools import reduce
from collections import OrderedDict

from gym.spaces import Box, Discrete, Dict, Tuple, MultiDiscrete


def num_envs(env):
    """
    <a href="https://github.com/seba-1511/cherry/blob/master/cherry/envs/utils.py" class="source-link">[Source]</a>

    ## Description

    Heuristic that returns the number of parallel workers in a vectorized environment.

    Warning: functionality is experimental, mostly tested with `gym.vector`.
    """
    if hasattr(env, 'num_envs'):
        return env.num_envs
    if hasattr(env, 'envs'):
        return len(env.envs)
    if hasattr(env, 'processes'):
        return len(env.processes)
    return 1


def is_vectorized(env):
    """
    <a href="https://github.com/seba-1511/cherry/blob/master/cherry/envs/utils.py" class="source-link">[Source]</a>

    ## Description

    Heuristic that returns whether an environment is vectorized or not.

    Warning: functionality is experimental, mostly tested with `gym.vector`.
    """
    return num_envs(env) > 1


def is_discrete(space, vectorized=False):
    """
    <a href="https://github.com/seba-1511/cherry/blob/master/cherry/envs/utils.py" class="source-link">[Source]</a>

    ## Description

    Returns whether a space is discrete.

    ## Arguments

    * `space` - The space.
    * `vectorized` - Whether to return the discreteness for the
        vectorized environments (True) or just the discreteness of
        the underlying environment (False).
    """
    msg = 'Space type not supported.'
    assert isinstance(space, (Box, Discrete, Dict, Tuple, MultiDiscrete)), msg
    if isinstance(space, (Discrete, MultiDiscrete)):
        return True
    if isinstance(space, Box):
        return False
    if isinstance(space, Dict):
        dimensions = {
            k[0]: is_discrete(k[1], vectorized) for k in space.spaces.items()
        }
        return OrderedDict(dimensions)
    if isinstance(space, Tuple):
        if not vectorized:
            return is_discrete(space[0], vectorized)
        discrete = tuple(
            is_discrete(s) for s in space
        )
        return discrete


def get_space_dimension(space, vectorized_dims=False):
    """
    <a href="https://github.com/seba-1511/cherry/blob/master/cherry/envs/utils.py" class="source-link">[Source]</a>

    ## Description

    Returns the number of elements of a space sample, when unrolled.

    ## Arguments

    * `space` - The space.
    * `vectorized_dims` - Whether to return the full dimension for vectorized
        environments (True) or just the dimension for the underlying
        environment (False).
    """
    msg = 'Space type not supported.'
    assert isinstance(space, (Box, Discrete, Dict, Tuple, MultiDiscrete)), msg
    if isinstance(space, Discrete):
        return space.n
    if isinstance(space, MultiDiscrete):
        if vectorized_dims:
            return reduce(operator.mul, space.nvec, 1)
        return int(space.nvec[0])
    if isinstance(space, Box):
        if len(space.shape) > 1 and not vectorized_dims:
            return reduce(operator.mul, space.shape[1:], 1)
        return reduce(operator.mul, space.shape, 1)
    if isinstance(space, Dict):
        dimensions = {
            k[0]: get_space_dimension(k[1], vectorized_dims) for k in space.spaces.items()
        }
        return OrderedDict(dimensions)
    if isinstance(space, Tuple):
        if not vectorized_dims:
            return get_space_dimension(space[0], vectorized_dims)
        dimensions = tuple(
            get_space_dimension(s) for s in space
        )
        return dimensions
