#!/usr/bin/env python3


def copy_params(src, trg):
    for s, t in zip(src, trg):
        t.data.copy_(s.data)


def dist_average(src, shared, weight, barrier, sync=True):
    updates = [weight * (s.data - sh.data) for s, sh, in zip(src, shared)]
    if sync:
        barrier.wait()
    for u, sh in zip(updates, shared):
        sh.data.add_(u)
    if sync:
        barrier.wait()
