#!/usr/bin/env python3

import torch as th
import numpy as np

from cherry._utils import EPS


def totensor(array, dtype=None):
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
    if dtype is None:
        dtype = th.get_default_dtype()
    if isinstance(array, int):
        array = float(array)
    if isinstance(array, float):
        array = [array, ]
    if isinstance(array, list):
        array = np.array(array)
    if isinstance(array, (np.ndarray,
                          np.bool_,
                          np.float32,
                          np.float64,
                          np.int32,
                          np.int64)):
        if array.dtype == np.bool_:
            array = array.astype(np.uint8)
        array = th.tensor(array, dtype=dtype)
        array = array.unsqueeze(0)
    return array


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
