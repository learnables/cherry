#!/usr/bin/env python3

import dataclasses
from collections.abc import Mapping


class AlgorithmArguments(Mapping):

    def __len__(self):
        return len(dataclasses.fields(self))

    def __iter__(self):
        for field in dataclasses.fields(self):
            yield field.name

    def __getitem__(self, item):
        return self.__dict__[item]
