#!/usr/bin/env python3

import dataclasses
import dotmap


class AlgorithmArguments:

    """
    Utility functions to work with dataclass algorithms.
    """

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
