#!/usr/bin/env python3

"""
**Description**

Optimization utilities for scalable, high-performance reinforcement learning.
"""

import torch.distributed as dist
from torch.optim.optimizer import Optimizer, required


class Distributed(Optimizer):

    """

    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/optim.py)

    **Description**

    Synchronizes the gradients of a model across replicas.

    At every step, `Distributed` averages the gradient across all replicas
    before calling the wrapped optimizer.
    The `sync` parameters determines how frequently the parameters are
    synchronized between replicas, to minimize numerical divergences.
    This is done by calling the `sync_parameters()` method.
    If `sync is None`, this never happens except upon initialization of the
    class.

    **Arguments**

    * **params** (iterable) - Iterable of parameters.
    * **opt** (Optimizer) - The optimizer to wrap and synchronize.
    * **sync** (int, *optional*, default=None) - Parameter
      synchronization frequency.

    **References**

    1. Zinkevich et al. 2010. “Parallelized Stochastic Gradient Descent.”

    **Example**

    ~~~python
    opt = optim.Adam(model.parameters())
    opt = Distributed(model.parameters(), opt, sync=1)

    opt.step()
    opt.sync_parameters()
    ~~~

    """

    def __init__(self, params=required, opt=required, sync=None):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.opt = opt
        self.sync = sync
        self.iter = 0
        defaults = {}
        super(Distributed, self).__init__(params, defaults)
        self.sync_parameters()

    def sync_parameters(self, root=0):
        """
        **Description**

        Broadcasts all parameters of root to all other replicas.

        **Arguments**

        * **root** (int, *optional*, default=0) - Rank of root replica.

        """
        if self.world_size > 1:
            for group in self.param_groups:
                for p in group['params']:
                    dist.broadcast(p.data, src=root)

    def step(self):
        if self.world_size > 1:
            num_replicas = float(self.world_size)
            # Average all gradients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad

                    # Perform the averaging
                    dist.all_reduce(d_p)
                    d_p.data.mul_(1.0 / num_replicas)

        # Perform optimization step
        self.opt.step()
        self.iter += 1

        if self.sync is not None and self.iter >= self.sync:
            self.sync_parameters()
            self.iter = 0
