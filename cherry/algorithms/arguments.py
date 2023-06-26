#!/usr/bin/env python3

import dataclasses
import dotmap
import collections


class AlgorithmArguments(collections.abc.Mapping):

    """
    <a href="https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/arguments.py" class="source-link">[Source]</a>

    ## Description

    Utility functions to work with dataclass algorithms.

    ## Example

    ~~~python
    @dataclasses.dataclass
    class MyNewAlgorithm(AlgorithmArguments).

        my_arg1: float = 0.0

        def update(self, my_arg1, **kwargs):
            pass
    ~~~
    """

    # turns algorithm arguments into a mapping
    def __len__(self):
        return len(dataclasses.fields(self))

    # turns algorithm arguments into a mapping
    def __iter__(self):
        for field in dataclasses.fields(self):
            yield field.name

    # turns algorithm arguments into a mapping
    def __getitem__(self, item):
        return self.__dict__[item]

    @staticmethod
    def unpack_config(obj, config):
        """
        Returns a DotMap, picking parameters first from config
        and if not present from obj.

        ## Arguments

        * `obj` (dataclass) - Algorithm to help fill missing values in config.
        * `config` (dict) - Partial configuration to get values from.
        """
        args = dotmap.DotMap()
        for field in dataclasses.fields(obj):
            args[field.name] = config.get(field.name, getattr(obj, field.name))
        return args
