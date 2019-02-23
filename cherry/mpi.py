#!/usr/bin/env python3

"""
Some utilities for using mpi4py with PyTorch tensors.
"""

import signal
import torch as th
from torch.optim.optimizer import Optimizer, required

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    CUDA_UNSUPPORTED = 'CUDA Tensors are currently unsupported.'

    # Allow for SIGINT to kill all child processes
    def terminate_mpi(sig=None, frame=None):
        comm.Abort()

    # This kills MPI if it is not used
    if size < 2:
        MPI.Finalize()
    else:
        signal.signal(signal.SIGINT, terminate_mpi)

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

    class Distributed(Optimizer):

        def __init__(self, params=required, opt=required):
            self.opt = opt
            defaults = {}
            super(Distributed, self).__init__(params, defaults)
            if size > 1:
                # Broadcast all parameters such that they are equal
                broadcast([p.data for p in params])

        def step(self):
            if size > 1:
                num_replicas = float(size)
                # Average all gradients
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        d_p = p.grad
                        assert not d_p.data.is_cuda, CUDA_UNSUPPORTED
                        param_state = self.state[p]

                        # Create buffer if necessary
                        if 'buffer' not in param_state:
                            param_state['buffer'] = th.zeros_like(d_p.data)
                        src_buffer = param_state['buffer']

                        # Perform the averaging
                        src_buffer.copy_(d_p.data)
                        comm.Allreduce(src_buffer.numpy(),
                                       d_p.data.numpy(),
                                       op=MPI.SUM)
                        d_p.data.mul_(1.0 / num_replicas)

            # Perform optimization step
            self.opt.step()


except ImportError:
    print('Warning: mpi4py not installed, ignoring.')
