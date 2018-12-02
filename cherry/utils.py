#!/usr/bin/env python3

import sys
import torch as th

EPS = sys.float_info.epsilon


def totensor(array):
    if not isinstance(array, th.Tensor):
        array = th.tensor(array)
        array = array.view(1, *array.size())
    return array


def normalize(tensor):
    return (tensor - tensor.mean()) / (tensor.std() + EPS)
