# -*- coding=utf-8 -*-

import torch


class Lambda(torch.nn.Module):

    """
    <a href="" class="source-link">[Source]</a>

    ## Description

    Turns any function into a PyTorch Module.

    ## Example

    ~~~python
    double = Lambda(lambda x: 2 * x)
    out = double(tensor([23]))  # out == 46
    ~~~
    """

    def __init__(self, fn):
        """
        ## Description

        * `fn` (callable) - Function to turn into a Module.
        """
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
