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
            self.replay.append(vector,
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
                self.replay.append(vector,
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
                self.replay.append(vector,
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
                self.replay.append(vector,
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
                self.replay.append(vector,
                                   vector,
                                   i,
                                   vector,
                                   random.choice([False, True]),
                                   info={'vector': vector, 'id': count})
            for _ in range(30):
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
                        if not sample[i].done:
                            self.assertEqual(sample[i]['info']['id']+1,
                                             sample[i+1]['info']['id'])

    def test_add(self):
        new_replay = ch.ExperienceReplay()
        vector = np.random.rand(VECTOR_SIZE)
        for i in range(NUM_SAMPLES):
            self.replay.append(vector,
                               vector,
                               i,
                               vector,
                               False,
                               info={'vector': vector})
            new_replay.append(vector,
                              vector,
                              i,
                              vector,
                              False,
                              info={'vector': vector})
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
                              info={'vector': vector})
        # save the old file
        old_replay.save('testing_temp_file.pt')

        # load the saved file to a new file
        new_replay = ch.ExperienceReplay()
        new_replay.load('testing_temp_file.pt')

        # check size
        self.assertEqual(len(old_replay.storage['states']),
                         len(new_replay.storage['states']))
        self.assertEqual(len(old_replay.storage['actions']),
                         len(new_replay.storage['actions']))
        self.assertEqual(len(old_replay.storage['rewards']),
                         len(new_replay.storage['rewards']))
        self.assertEqual(len(old_replay.storage['next_states']),
                         len(new_replay.storage['next_states']))
        self.assertEqual(len(old_replay.storage['dones']),
                         len(new_replay.storage['dones']))
        self.assertEqual(len(old_replay.storage['infos']),
                         len(new_replay.storage['infos']))

        # check content
        for a, b in zip(old_replay, new_replay):
            self.assertTrue(close(a.state, b.state))
            self.assertTrue(close(a.action, b.action))
            self.assertTrue(close(a.reward, b.reward))
            self.assertTrue(close(a.next_state, b.next_state))
            self.assertTrue(close(a.done, b.done))
            self.assertEqual(a.info['vector'].all(),
                             b.info['vector'].all())

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
                                   info={'test': test_tensor})
        self.assertTrue(isinstance(standard_replay.tests, th.Tensor))

        # stuff in an int, the result type should be a list
        put_int_replay = standard_replay[:]
        put_int_replay.append(vector,
                              vector,
                              i,
                              vector,
                              False,
                           info={'test': 1000})
        self.assertTrue(isinstance(put_int_replay.tests, list))

        # stuff in a float, the result type should be a list
        put_float_replay = standard_replay[:]
        put_float_replay.append(vector,
                                vector,
                                i,
                                vector,
                                False,
                                info={'test': float(9.8981)})
        self.assertTrue(isinstance(put_float_replay.tests, list))

    def test_slices(self):
        for i in range(NUM_SAMPLES):
            self.replay.append(th.randn(VECTOR_SIZE),
                               th.randn(VECTOR_SIZE),
                               i,
                               th.randn(VECTOR_SIZE),
                               False,
                               info={'vector': th.randn(VECTOR_SIZE)})

        sliced = self.replay[0:-3]
        self.assertEqual(len(sliced), len(self.replay) - 3)
        for sars, sars_ in zip(self.replay, sliced):
            self.assertTrue(close(sars.state, sars_.state))
            self.assertTrue(close(sars.action, sars_.action))
            self.assertTrue(close(sars.reward, sars_.reward))
            self.assertTrue(close(sars.next_state, sars_.next_state))
            self.assertTrue(close(sars.info['vector'], sars_.info['vector']))

    def update_test(self):
        for i in range(NUM_SAMPLES):
            self.replay.append(th.randn(VECTOR_SIZE),
                               th.randn(VECTOR_SIZE),
                               i,
                               th.randn(VECTOR_SIZE),
                               False,
                               info={'vector': th.randn(VECTOR_SIZE)})

        clone = ch.ExperienceReplay(states=self.replay.states.clone().detach(),
                                    actions=self.replay.actions.clone().detach(),
                                    rewards=self.replay.rewards.clone().detach(),
                                    next_states=self.replay.next_states.clone().detach(),
                                    infos=copy.deepcopy(self.replay.infos))
        self.replay.update(lambda i, sars: {
            'reward': sars.reward + 1,
            'action': sars.action + 1,
            'state': sars.state + 1,
            'infos': {
                'vector': sars.info['vector'] + 1,
            }
        })


if __name__ == '__main__':
    unittest.main()
