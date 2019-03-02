#!/usr/bin/env python3


def polyak_average(source, target, alpha):
    """
    """
    for s, t in zip(source.parameters(), target.parameters()):
        s.data.mul_(1.0 - alpha).add_(alpha, t.data)
