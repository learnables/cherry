#!/usr/bin/env python3

import unittest
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

