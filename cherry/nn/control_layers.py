#!/usr/bin/env python3

import torch.nn as nn
import cherry as ch


class ControlLinear(nn.Linear):

    def __init__(self, *args, **kwargs):
        gain = kwargs.pop('gain', None)
        super(ControlLinear, self).__init__(*args, **kwargs)
        ch.nn.init.kostrikov_control_(self, gain=gain)
