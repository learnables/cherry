#!/usr/bin/env python3

import random
import unittest
import numpy as np
import torch as th
import cherry as ch
import cherry.envs as envs

import gym
from gym.vector import AsyncVectorEnv
from gym.envs.unittest import MemorizeDigits

#from dummy_env import Dummy


NUM_STEPS = 10


class Dummy(gym.Env):

    """
    A dummy environment that returns random states and rewards.
    """

    def __init__(self):
        low = np.array([-5, -5, -5, -5, -5])
        high = -np.array([-5, -5, -5, -5, -5])
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.rng = random.Random()

    def step(self, action):
        assert self.observation_space.contains(action)
        next_state = self.observation_space.sample()
        reward = action.sum()
        done = random.random() > 0.95
        info = {}
        return next_state, reward, done, info

    def reset(self):
        return self.observation_space.sample()

    def seed(self, seed=1234):
        self.rng.seed(seed)
        np.random.seed(seed)

    def _render(self, mode='human', close=False):
        pass

    def _take_action(self, action):
        pass

    def _get_reward(self):
        return self.rng.randint(0, 10)


class TestRunnerWrapper(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_vec_env(self):
        """
        TODO:
        - policy return a list of torch vectors
        - policy return a single large tensor
        - policy return (tensor, info)
        - policy return ([tensors], info)
        - policy return (ndarray, info)
        - policy return ([ndarry], info)
        - Using Logger
        - For all cases above: .state(), .done(), .action()
        - Try different envs (not just Dummy)
        - Try 1, 2, 4 envs
        - Check expected shapes of each property
        - Use AsyncVec vs gym.vector.make
        """
#        n_envs = 4
#        def make_env():
#            env = Dummy()
#            return env
#        env_fns = [make_env for _ in range(n_envs)]
#        env = async_env = AsyncVectorEnv(env_fns)
##        env = envs.Torch(env)
#        env = envs.Runner(env)
#
#        policy = lambda x: (env.action_space.sample(), {})
#        replay = env.run(policy, steps=10)
#
#        policy = lambda x: env.action_space.sample()
#        replay = env.run(policy, steps=10)

        def test_config(n_envs,
                        base_env,
                        use_torch,
                        use_logger,
                        return_info):
            config = 'n_envs' + str(n_envs) + '-base_env' + str(base_env) \
                    + '-torch' + str(use_torch) + '-logger' + str(use_logger) \
                    + '-info' + str(return_info)
#            print(config)
            if isinstance(base_env, str):
                env = vec_env = gym.vector.make(base_env, num_envs=n_envs)
            else:
                def make_env():
                    env = base_env()
                    return env
                env_fns = [make_env for _ in range(n_envs)]
                env = vec_env = AsyncVectorEnv(env_fns)

            if use_logger:
#                env = envs.Logger(env, interval=5)
                env = envs.Logger(env)

            if use_torch:
                env = envs.Torch(env)
                policy = lambda x: ch.totensor(vec_env.action_space.sample())
            else:
                policy = lambda x: vec_env.action_space.sample()

            if return_info:
                agent = lambda x: (policy(x), {'policy': policy(x)[0]})
            else:
                agent = policy

            # Gather experience
            env = envs.Runner(env)
            replay = env.run(agent, steps=NUM_STEPS)

            # Pre-compute some shapes
            shape = (NUM_STEPS, n_envs)
            state_shape = vec_env.observation_space.sample()[0]
            if isinstance(state_shape, (int, float)):
                state_shape = tuple()
            else:
                state_shape = state_shape.shape
            action_shape = vec_env.action_space.sample()[0]
            if isinstance(action_shape, (int, float)):
                action_shape = (1, )
            else:
                action_shape = action_shape.shape
            done_shape = tuple()

            # Check shapes
            states = replay.state()
            self.assertEqual(states.shape, shape + state_shape, config)
            actions = replay.action()
            self.assertEqual(actions.shape, shape + action_shape, config)
            dones = replay.done()
            self.assertEqual(dones.shape, shape + done_shape, config)
            if return_info:
                policies = replay.policy()
                self.assertEqual(policies.shape, (NUM_STEPS, ) + action_shape, config)

        test_config(2, Dummy, True, True, True)


        for return_info in [False, True]:
            for use_logger in [False, True]:
                for use_torch in [False, True]:
                    for base_env in [Dummy, MemorizeDigits, 'MemorizeDigits-v0']:
                        for n_envs in [2, 4]:
                            test_config(n_envs,
                                        base_env,
                                        use_torch,
                                        use_logger,
                                        return_info)
                                


if __name__ == '__main__':
    unittest.main()
