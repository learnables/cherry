#!/usr/bin/env python3

"""
Some utilities for using mpi4py with PyTorch.
"""

import torch as th
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def allreduce(tensors, op='mean'):
    # TODO: Handle GPU tensors
    if isinstance(tensors, th.Tensor):
        tensors = [tensors]

    mpi_op = MPI.SUM
    if op == 'product':
        mpi_op = MPI.PROD

    for tensor in tensors:
        target = tensor.data.numpy()
        source = target.copy()
        comm.Allreduce(source, target, op=mpi_op)
        if op == 'mean':
            source /= float(size)
