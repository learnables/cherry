#!/usr/bin/env python3

import sys
import numpy as np
import torch as th

from gym.spaces import Box, Discrete


EPS = sys.float_info.epsilon


def totensor(array):
    if not isinstance(array, th.Tensor):
        array = th.tensor(array)
        array = array.view(1, *array.size())
    return array


def normalize(tensor):
    return (tensor - tensor.mean()) / (tensor.std() + EPS)


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
