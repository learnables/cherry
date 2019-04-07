#!/usr/bin/env python3

import numpy as np
import torch as th

EPS = 1e-8


def totensor(array, dtype=None):
    if dtype is None:
        dtype = th.get_default_dtype()
    if isinstance(array, int):
        array = float(array)
    if isinstance(array, float):
        array = [array, ]
    if isinstance(array, list):
        array = np.array(array)
    if isinstance(array, (np.ndarray, np.bool_)):
        if array.dtype == np.bool_:
            array = array.astype(np.uint8)
        array = th.tensor(array, dtype=dtype)
        array = array.unsqueeze(0)
    return array


def min_size(tensor):
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


def normalize(tensor, epsilon=EPS):
    """
    [[Source]]()

    **Description**

    Normalizes a tensor to zero mean and unit std.

    **References**

    **Arguments**

    **Returns**

    **Example**

    """
    if tensor.numel() <= 1:
        return tensor
    return (tensor - tensor.mean()) / (tensor.std() + epsilon)


def onehot(x, dim):
    """
    [[Source]]()

    **Description**

    Creates a new onehot encoded tensor.

    **References**

    **Arguments**

    **Returns**

    **Example**

    """
    size = 1
    if isinstance(x, np.ndarray):
        size = x.shape[0]
        x = th.from_numpy(x).long()
    if isinstance(x, th.Tensor):
        size = x.size(0)
        x = x.long()
    onehot = th.zeros(size, dim)
    onehot[:, x] = 1.0
    return onehot


class _ImportRaiser(object):

    def __init__(self, name, command):
        self.name = name
        self.command = command

    def __getattr__(self, *args, **kwargs):
        msg = self.name + ' required. Try: ' + self.command
        raise ImportError(msg)
