#!/usr/bin/env python3

import sys
import numpy as np
import torch as th
import operator

from functools import reduce
from collections import OrderedDict

from gym.spaces import Box, Discrete, Dict


EPS = sys.float_info.epsilon


def totensor(array):
    if isinstance(array, (int, float)):
        array = [array, ]
    if isinstance(array, list):
        array = np.array(array, dtype=np.float32)
    if isinstance(array, (np.ndarray, np.bool_)):
        if array.dtype == np.bool_:
            array = array.astype(np.uint8)
        array = th.tensor(array)
        array = array.view(1, *array.size())
    return array


def min_size(tensor):
    true_size = tensor.size()
    if len(true_size) < 1:
        return (1, )
    while true_size[0] == 1 and len(true_size) > 1:
        true_size = true_size[1:]
    return true_size


def normalize(tensor, epsilon=EPS):
    if tensor.numel() <= 1:
        return tensor
    return (tensor - tensor.mean()) / (tensor.std() + epsilon)


def onehot(x, dim):
    onehot = np.zeros(1, dim)
    onehot[x] = 1.0
    return onehot


def flatten_state(space, state):
    if isinstance(space, Box):
        return np.asarray(state).flatten()
    if isinstance(space, Discrete):
        return onehot(state, space.n)
    raise('The space was not recognized.')


def get_space_dimension(space):
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
