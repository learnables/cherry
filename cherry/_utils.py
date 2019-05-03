#!/usr/bin/env python3

import numpy as np
import torch as th

EPS = 1e-8


def _reshape_helper(tensor):
    if len(tensor.size()) == 1:
        return tensor.view(-1, 1)
    return tensor


def _istensorable(array):
    types = (int,
             float,
             list,
             np.ndarray,
             np.bool_,
             th.Tensor)
    if isinstance(array, types):
        return True
    return False


def _min_size(tensor):
    """
    [[Source]]()

    **Description**

    Returns the minimium viewable size of a tensor.
    e.g. (1, 1, 3, 4) -> (3, 4)

    **References**

    **Arguments**

    **Returns**

    **Example**

    """
    true_size = tensor.size()
    if len(true_size) < 1:
        return (1, )
    while true_size[0] == 1 and len(true_size) > 1:
        true_size = true_size[1:]
    return true_size


class _ImportRaiser(object):

    def __init__(self, name, command):
        self.name = name
        self.command = command

    def __getattr__(self, *args, **kwargs):
        msg = self.name + ' required. Try: ' + self.command
        raise ImportError(msg)
