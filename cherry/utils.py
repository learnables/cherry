#!/usr/bin/env python3

import numpy as np
import torch as th

EPS = np.finfo(np.float32).eps.item()


def totensor(array):
    if not isinstance(array, th.Tensor):
        array = th.tensor(array)
        array = array.view(1, *array.size())
    return array


def normalize(tensor):
    return (tensor - tensor.mean()) / (tensor.std() + EPS)
