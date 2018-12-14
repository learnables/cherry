#!/usr/bin/env python3

import random
import unittest
import numpy as np
import torch as th
import cherry as ch
import cherry.envs as envs

from dummy_env import Dummy


NUM_STEPS = 10000


class TestLoggerWrapper(unittest.TestCase):

    def setUp(self):
        env = Dummy()
        self.logger = envs.Logger(env)

    def tearDown(self):
        pass

    def test_custom_values(self):
        values = {
            'entropy': [],
            'policy_loss': [],
            'value_loss': [],
        }
        for step in range(NUM_STEPS):
            for key in ['entropy', 'policy_loss', 'value_loss']:
                value = random.randint(0, 10)
                self.logger.log(key, value)
                values[key].append(value)
        for i in range(NUM_STEPS):
            for key in values.keys():
                self.assertEqual(getattr(self.logger, key)[i], values[key][i])
