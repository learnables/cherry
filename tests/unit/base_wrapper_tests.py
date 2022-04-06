#!/usr/bin/env python3

import cherry
import gym
import operator
import unittest

from functools import reduce
from gym.spaces import Box, Discrete


def ref_space_dims(space):
    if isinstance(space, Discrete):
        return space.n
    if isinstance(space, Box):
        return reduce(operator.mul, space.shape, 1)


class TestBaseWrapper(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sizes(self):
        for env_name in ['CartPole-v1', 'Pendulum-v1']:
            ref_env = gym.make(env_name)
            cherry_env = cherry.envs.Torch(ref_env)

            # Test base implementation
            ref_action_size = cherry_env.action_size
            ref_state_size = cherry_env.state_size
            self.assertTrue(isinstance(ref_action_size, int))
            self.assertTrue(isinstance(ref_state_size, int))
            self.assertEqual(ref_action_size, ref_space_dims(ref_env.action_space))
            self.assertEqual(ref_state_size, ref_space_dims(ref_env.observation_space))

            ref_state_shape = (1,) + ref_env.reset().shape
            cherry_state_shape = cherry_env.reset().shape
            self.assertEqual(ref_state_shape, cherry_state_shape)

            # Test vectorized environments
            for num_envs in range(1, 4):
                vec_env = gym.vector.make(env_name, num_envs=num_envs)
                cherry_env = cherry.envs.Torch(vec_env)
                self.assertTrue(isinstance(cherry_env.state_size, int))
                self.assertTrue(isinstance(cherry_env.action_size, int))
                self.assertEqual(cherry_env.state_size, ref_state_size)
                self.assertEqual(cherry_env.action_size, ref_action_size)

                ref_state_shape = (num_envs,) + ref_env.reset().shape
                cherry_state_shape = cherry_env.reset().shape
                self.assertEqual(ref_state_shape, cherry_state_shape)


if __name__ == '__main__':
    unittest.main()
