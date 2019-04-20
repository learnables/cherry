#!/usr/bin/env python3

import unittest
import random
import numpy as np
import torch as th
import cherry as ch
import os
import copy


NUM_SAMPLES = 100
VECTOR_SIZE = 5


def close(a, b):
    return (a-b).norm(p=2) <= 1e-8


class TestExperienceReplay(unittest.TestCase):

    def setUp(self):
        self.replay = ch.ExperienceReplay()

    def tearDown(self):
        pass

    def test_empty(self):
        vector = np.random.rand(VECTOR_SIZE)
        for i in range(NUM_SAMPLES):
            self.replay.append(vector,
                               vector,
                               i,
                               vector,
                               False,
                               vector=vector)
        self.replay.empty()
        self.assertEqual(len(self.replay._storage), 0)

    def test_len(self):
        vector = np.random.rand(VECTOR_SIZE)
        for i in range(NUM_SAMPLES):
            self.replay.append(vector,
                               vector,
                               i,
                               vector,
                               False,
                               vector=vector)
        self.assertEqual(len(self.replay), NUM_SAMPLES)
        self.assertEqual(len(self.replay._storage), NUM_SAMPLES)

    def test_add_numpy(self):
        for shape in [(VECTOR_SIZE,), (1, VECTOR_SIZE)]:
            vector = np.random.rand(*shape)
            for i in range(NUM_SAMPLES):
                self.replay.append(vector,
                                   vector,
                                   i,
                                   vector,
                                   False,
                                   vector=vector)
            ref_size = th.Size([NUM_SAMPLES, VECTOR_SIZE])
            self.assertTrue(isinstance(self.replay.state(), th.Tensor))
            self.assertEqual(self.replay.state().size(), ref_size)
            self.assertTrue(isinstance(self.replay.action(), th.Tensor))
            self.assertEqual(self.replay.action().size(), ref_size)
            self.assertTrue(isinstance(self.replay.reward(), th.Tensor))
            self.assertTrue(isinstance(self.replay.next_state(), th.Tensor))
            self.assertEqual(self.replay.next_state().size(), ref_size)
            self.assertTrue(isinstance(self.replay.done(), th.Tensor))
            self.replay.empty()

    def test_add_torch(self):
        for shape in [(VECTOR_SIZE, ), (1, VECTOR_SIZE)]:
            vector = th.randn(*shape)
            for i in range(NUM_SAMPLES):
                self.replay.append(vector,
                                   vector,
                                   i,
                                   vector,
                                   False,
                                   vector=vector)
            ref_size = th.Size([NUM_SAMPLES, VECTOR_SIZE])
            self.assertTrue(isinstance(self.replay.state(), th.Tensor))
            self.assertEqual(self.replay.state().size(), ref_size)
            self.assertTrue(isinstance(self.replay.action(), th.Tensor))
            self.assertEqual(self.replay.action().size(), ref_size)
            self.assertTrue(isinstance(self.replay.reward(), th.Tensor))
            self.assertTrue(isinstance(self.replay.next_state(), th.Tensor))
            self.assertEqual(self.replay.next_state().size(), ref_size)
            self.assertTrue(isinstance(self.replay.done(), th.Tensor))
            self.replay.empty()

    def test_slice(self):
        # Fill replay
        count = 0
        for shape in [(VECTOR_SIZE, ), (1, VECTOR_SIZE)]:
            vector = th.randn(*shape)
            for i in range(NUM_SAMPLES):
                count += 1
                self.replay.append(vector,
                                   vector,
                                   i,
                                   vector,
                                   random.choice([False, True]),
                                   vector=vector,
                                   id=count)

        subsample = self.replay[0:len(self.replay)//2]
        self.assertTrue(isinstance(subsample, ch.ExperienceReplay))
        self.assertEqual(len(subsample), len(self.replay)//2)
        self.assertTrue(isinstance(self.replay[0], ch.Transition))
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
                self.replay.append(vector,
                                   vector,
                                   i,
                                   vector,
                                   random.choice([False, True]),
                                   vector=vector,
                                   id=count)
            for _ in range(30):
                # Test default arguments
                sample = self.replay.sample()
                self.assertEqual(len(sample), 1)
                # Test size
                sample = self.replay.sample(size=NUM_SAMPLES//2)
                self.assertEqual(len(sample), NUM_SAMPLES//2)
                # Test contiguous
                sample = self.replay.sample(size=NUM_SAMPLES//3,
                                            contiguous=True)
                ids = sample.id()
                for i, id in enumerate(ids[:-1]):
                    self.assertEqual(id + 1, ids[i+1])

                # Test single episode
                sample = self.replay.sample(size=1, episodes=True)
                self.assertTrue(bool(sample[-1].done.item()))
                for i, sars in enumerate(sample[:-1]):
                    self.assertTrue(not sample[i].done)
                    self.assertEqual(sars.id + 1,
                                     sample[i+1].id)

                # Test multiple episodes
                total_episodes = self.replay.done().sum().int().item()
                for num_episodes in [total_episodes, total_episodes//2, 1]:
                    sample = self.replay.sample(size=num_episodes,
                                                episodes=True)
                    num_sampled_episodes = sample.done().sum().int().item()
                    self.assertEqual(num_sampled_episodes, num_episodes)

                # Test multiple contiguous episodes
                total_episodes = self.replay.done().sum().int().item()
                for num_episodes in [total_episodes, total_episodes//2, 1]:
                    sample = self.replay.sample(size=num_episodes,
                                                episodes=True)
                    num_sampled_episodes = sample.done().sum().int().item()
                    self.assertEqual(num_sampled_episodes, num_episodes)
                    for i, sars in enumerate(sample[:-1]):
                        if not sars.done:
                            self.assertEqual(sample[i].id+1,
                                             sample[i+1].id)

    def test_append(self):
        new_replay = ch.ExperienceReplay()
        vector = np.random.rand(VECTOR_SIZE)
        for i in range(NUM_SAMPLES):
            self.replay.append(vector,
                               vector,
                               i,
                               vector,
                               False,
                               vector=vector)
            new_replay.append(vector,
                              vector,
                              i,
                              vector,
                              False,
                              vector=vector)
        self.assertEqual(len(self.replay), len(new_replay))
        new_replay = self.replay + new_replay
        self.assertEqual(NUM_SAMPLES * 2, len(new_replay))
        self.replay += new_replay
        self.assertEqual(NUM_SAMPLES * 3, len(self.replay))

    def test_save_and_load(self):
        old_replay = self.replay
        vector = np.random.rand(VECTOR_SIZE)
        for i in range(NUM_SAMPLES):
            old_replay.append(vector,
                              vector,
                              i,
                              vector,
                              False,
                              vector=vector)
        # save the old file
        old_replay.save('testing_temp_file.pt')

        # load the saved file to a new file
        new_replay = ch.ExperienceReplay()
        new_replay.load('testing_temp_file.pt')

        # check size
        self.assertEqual(len(old_replay._storage),
                         len(new_replay._storage))
        self.assertEqual(len(old_replay.state()),
                         len(new_replay.state()))
        self.assertEqual(len(old_replay.action()),
                         len(new_replay.action()))
        self.assertEqual(len(old_replay.reward()),
                         len(new_replay.reward()))
        self.assertEqual(len(old_replay.next_state()),
                         len(new_replay.next_state()))
        self.assertEqual(len(old_replay.done()),
                         len(new_replay.done()))
        self.assertEqual(len(old_replay.vector()),
                         len(new_replay.vector()))

        # check content
        for a, b in zip(old_replay, new_replay):
            self.assertTrue(close(a.state, b.state))
            self.assertTrue(close(a.action, b.action))
            self.assertTrue(close(a.reward, b.reward))
            self.assertTrue(close(a.next_state, b.next_state))
            self.assertTrue(close(a.done, b.done))
            self.assertTrue(close(a.vector, b.vector))

        os.remove('testing_temp_file.pt')

    def test_replay_myattr(self):
        standard_replay = self.replay
        vector = np.random.rand(VECTOR_SIZE)

        # a random tensor to be stuffed in
        test_tensor = th.randn(3, 3, dtype=th.double)

        # initialization, stuff just tensors in
        # and the results type should still be tensor
        for i in range(NUM_SAMPLES):
            standard_replay.append(vector,
                                   vector,
                                   i,
                                   vector,
                                   False,
                                   test=test_tensor)
        self.assertTrue(isinstance(standard_replay.test(), th.Tensor))

    def test_slices(self):
        for i in range(NUM_SAMPLES):
            self.replay.append(th.randn(VECTOR_SIZE),
                               th.randn(VECTOR_SIZE),
                               i,
                               th.randn(VECTOR_SIZE),
                               False,
                               vector=th.randn(VECTOR_SIZE))

        sliced = self.replay[0:-3]
        self.assertEqual(len(sliced), len(self.replay) - 3)
        for sars, sars_ in zip(self.replay, sliced):
            self.assertTrue(close(sars.state, sars_.state))
            self.assertTrue(close(sars.action, sars_.action))
            self.assertTrue(close(sars.reward, sars_.reward))
            self.assertTrue(close(sars.next_state, sars_.next_state))
            self.assertTrue(close(sars.vector, sars_.vector))

    def update_test(self):
        for i in range(NUM_SAMPLES):
            self.replay.append(th.randn(VECTOR_SIZE),
                               th.randn(VECTOR_SIZE),
                               i,
                               th.randn(VECTOR_SIZE),
                               False,
                               vector=th.randn(VECTOR_SIZE))

        clone = ch.ExperienceReplay(copy.deepcopy(self.replay._storage))
        self.replay.update(lambda i, sars: {
            'reward': sars.reward + 1,
            'action': sars.action + 1,
            'state': sars.state + 1,
            'vector': sars.vector + 1,
        })
        for sars, sars_ in zip(clone, self.replay):
            self.assertTrue(close(sars.reward + 1, sars_.reward))
            self.assertTrue(close(sars.action + 1, sars_.action))
            self.assertTrue(close(sars.state + 1, sars_.state))
            self.assertTrue(close(sars.vector + 1, sars_.vector))


if __name__ == '__main__':
    unittest.main()
