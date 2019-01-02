#!/usr/bin/env python3

"""
Some utilities for using mpi4py with PyTorch tensors.
"""

import torch as th
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def broadcast(tensors, root=0):
    # TODO: Handle GPU tensors
    if isinstance(tensors, th.Tensor):
        tensors = [tensors]
    for tensor in tensors:
        comm.Bcast(tensor.numpy(), root)


def allreduce(tensors, op='mean'):
    # TODO: Handle GPU tensors
    if isinstance(tensors, th.Tensor):
        tensors = [tensors]

    mpi_op = MPI.SUM
    if op == 'product':
        mpi_op = MPI.PROD
    if op == 'min':
        mpi_op = MPI.MIN
    if op == 'max':
        mpi_op = MPI.MAX

    for tensor in tensors:
        target = tensor.numpy()
        source = target.copy()
        comm.Allreduce(source, target, op=mpi_op)
        if op == 'mean':
            source /= float(size)
