#!/usr/bin/env python3

import unittest
import cherry as ch

from dummy_env import Dummy


class TestModelsUtils(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_random_policy(self):
        env = Dummy()
        env = ch.envs.Runner(env)
        policy = ch.models.RandomPolicy(env)
        env.run(policy, episodes=10)


if __name__ == '__main__':
    unittest.main()
