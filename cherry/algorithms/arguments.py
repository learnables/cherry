#!/usr/bin/env python3

import dataclasses
import dotmap

from collections import Mapping


@dataclasses.dataclass
class AlgorithmArguments(Mapping):

    # *** Turns AlgorithmArguments instances into dataclasses **

    def __len__(self):
        return len(dataclasses.fields(self))

    def __iter__(self):
        for field in dataclasses.fields(self):
            yield field.name

    def __getitem__(self, item):
        return self.__dict__[item]

    @staticmethod
    def unpack_config(obj, config):
        """
        Returns a DotMap, picking parameters first from config
        and if not present from obj.
        """
        args = dotmap.DotMap()
        for name in obj:
            args[name] = config.get(name, obj[name])
        return args
