#!/usr/bin/env python3

import torch.distributed as dist
from torch.optim.optimizer import Optimizer, required


class Distributed(Optimizer):

    def __init__(self, params=required, opt=required):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.opt = opt
        defaults = {}
        super(Distributed, self).__init__(params, defaults)
        if self.world_size > 1:
            # Broadcast all parameters such that they are equal
            for p in params:
                dist.broadcast(p.data, src=0)

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
