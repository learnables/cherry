#!/usr/bin/env python3

import unittest
import random
import numpy as np
import torch as th
import cherry as ch


NUM_SAMPLES = 100
VECTOR_SIZE = 5


class TestExperienceReplay(unittest.TestCase):

    def setUp(self):
        self.replay = ch.ExperienceReplay()

    def tearDown(self):
        pass

    def test_empty(self):
        vector = np.random.rand(VECTOR_SIZE)
        for i in range(NUM_SAMPLES):
            self.replay.add(vector,
                            vector,
                            i,
                            vector,
                            False,
                            info={'vector': vector})
        self.replay.empty()
        self.assertEqual(len(self.replay.storage['states']), 0)
        self.assertEqual(len(self.replay.storage['actions']), 0)
        self.assertEqual(len(self.replay.storage['rewards']), 0)
        self.assertEqual(len(self.replay.storage['next_states']), 0)
        self.assertEqual(len(self.replay.storage['dones']), 0)
        self.assertEqual(len(self.replay.storage['infos']), 0)

    def test_len(self):
        vector = np.random.rand(VECTOR_SIZE)
        for i in range(NUM_SAMPLES):
            self.replay.add(vector,
                            vector,
                            i,
                            vector,
                            False,
                            info={'vector': vector})
        self.assertEqual(len(self.replay), NUM_SAMPLES)
        self.assertEqual(len(self.replay.storage['states']), NUM_SAMPLES)
        self.assertEqual(len(self.replay.storage['actions']), NUM_SAMPLES)
        self.assertEqual(len(self.replay.storage['rewards']), NUM_SAMPLES)
        self.assertEqual(len(self.replay.storage['next_states']), NUM_SAMPLES)
        self.assertEqual(len(self.replay.storage['dones']), NUM_SAMPLES)
        self.assertEqual(len(self.replay.storage['infos']), NUM_SAMPLES)

    def test_add_numpy(self):
        for shape in [(VECTOR_SIZE,), (1, VECTOR_SIZE)]:
            vector = np.random.rand(*shape)
            for i in range(NUM_SAMPLES):
                self.replay.add(vector,
                                vector,
                                i,
                                vector,
                                False,
                                info={'vector': vector})
            ref_size = th.Size([NUM_SAMPLES, VECTOR_SIZE])
            self.assertTrue(isinstance(self.replay.states, th.Tensor))
            self.assertEqual(self.replay.states.size(), ref_size)
            self.assertTrue(isinstance(self.replay.actions, th.Tensor))
            self.assertEqual(self.replay.actions.size(), ref_size)
            self.assertTrue(isinstance(self.replay.rewards, th.Tensor))
            self.assertTrue(isinstance(self.replay.next_states, th.Tensor))
            self.assertEqual(self.replay.next_states.size(), ref_size)
            self.assertTrue(isinstance(self.replay.dones, th.Tensor))
            self.replay.empty()

    def test_add_torch(self):
        for shape in [(VECTOR_SIZE, ), (1, VECTOR_SIZE)]:
            vector = th.randn(*shape)
            for i in range(NUM_SAMPLES):
                self.replay.add(vector,
                                vector,
                                i,
                                vector,
                                False,
                                info={'vector': vector})
            ref_size = th.Size([NUM_SAMPLES, VECTOR_SIZE])
            self.assertTrue(isinstance(self.replay.states, th.Tensor))
            self.assertEqual(self.replay.states.size(), ref_size)
            self.assertTrue(isinstance(self.replay.actions, th.Tensor))
            self.assertEqual(self.replay.actions.size(), ref_size)
            self.assertTrue(isinstance(self.replay.rewards, th.Tensor))
            self.assertTrue(isinstance(self.replay.next_states, th.Tensor))
            self.assertEqual(self.replay.next_states.size(), ref_size)
            self.assertTrue(isinstance(self.replay.dones, th.Tensor))
            self.replay.empty()

    def test_slice(self):
        # Fill replay
        count = 0
        for shape in [(VECTOR_SIZE, ), (1, VECTOR_SIZE)]:
            vector = th.randn(*shape)
            for i in range(NUM_SAMPLES):
                count += 1
                self.replay.add(vector,
                                vector,
                                i,
                                vector,
                                random.choice([False, True]),
                                info={'vector': vector, 'id': count})

        subsample = self.replay[0:len(self.replay)//2]
        self.assertTrue(isinstance(subsample, ch.ExperienceReplay))
        self.assertEqual(len(subsample), len(self.replay)//2)
        self.assertTrue(isinstance(self.replay[0], dict))
        subsample = self.replay[-1:]
        self.assertEqual(len(subsample), 1)

    def test_sample(self):
        # Test empty
        sample = self.replay.sample()
        self.assertEqual(len(sample), 0)
        self.assertTrue(isinstance(self.replay, ch.ExperienceReplay))
        # Fill replay
        count = 0
        for shape in [(VECTOR_SIZE, ), (1, VECTOR_SIZE)]:
            vector = th.randn(*shape)
            for i in range(NUM_SAMPLES):
                count += 1
                self.replay.add(vector,
                                vector,
                                i,
                                vector,
                                random.choice([False, True]),
                                info={'vector': vector, 'id': count})
            for _ in range(50):
                # Test default arguments
                sample = self.replay.sample()
                self.assertEqual(len(sample), 1)
                # Test size
                sample = self.replay.sample(size=NUM_SAMPLES//2)
                self.assertEqual(len(sample), NUM_SAMPLES//2)
                # Test contiguous
                sample = self.replay.sample(size=NUM_SAMPLES//3, contiguous=True)
                infos = sample.infos
                for i in range(len(infos)-1):
                    self.assertEqual(infos[i]['id'] + 1, infos[i+1]['id'])
                # Test single episode
                sample = self.replay.sample(size=1, episodes=True)
                self.assertTrue(bool(sample[-1]['done'].item()))
                for i, sars in enumerate(sample[:-1]):
                    self.assertTrue(not sample[i].done)
                    self.assertEqual(sample[i]['info']['id']+1,
                                     sample[i+1]['info']['id'])

                # Test multiple episodes
                total_episodes = self.replay.dones.sum().int().item()
                for num_episodes in [total_episodes, total_episodes//2, 1]:
                    sample = self.replay.sample(size=num_episodes, episodes=True)
                    num_sampled_episodes = sample.dones.sum().int().item()
                    self.assertEqual(num_sampled_episodes, num_episodes)

                # Test multiple contiguous episodes
                total_episodes = self.replay.dones.sum().int().item()
                for num_episodes in [total_episodes, total_episodes//2, 1]:
                    sample = self.replay.sample(size=num_episodes, episodes=True)
                    num_sampled_episodes = sample.dones.sum().int().item()
                    self.assertEqual(num_sampled_episodes, num_episodes)
                    for i, sars in enumerate(sample[:-1]):
                        self.assertTrue(not sample[i].done)
                        self.assertEqual(sample[i]['info']['id']+1,
                                         sample[i+1]['info']['id'])
