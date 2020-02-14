#!/usr/bin/env python3

import unittest
import numpy as np
import torch as th
import cherry as ch

DIM = 5


class TestTorch(unittest.TestCase):

    def test_onehot(self):
        single = 3
        oh_single = ch.onehot(single, dim=DIM)
        self.assertTrue(oh_single.size(0) == 1)
        self.assertTrue(oh_single.size(1) == DIM)
        ref = th.zeros(1, DIM)
        ref[0, single] += 1
        self.assertTrue((oh_single - ref).pow(2).sum().item() == 0)

        single = np.int64(3)
        oh_single = ch.onehot(single, dim=DIM)
        self.assertTrue(oh_single.size(0) == 1)
        self.assertTrue(oh_single.size(1) == DIM)
        ref = th.zeros(1, DIM)
        ref[0, single] += 1
        self.assertTrue((oh_single - ref).pow(2).sum().item() == 0)

        single = np.float64(3.0)
        oh_single = ch.onehot(single, dim=DIM)
        self.assertTrue(oh_single.size(0) == 1)
        self.assertTrue(oh_single.size(1) == DIM)
        ref = th.zeros(1, DIM)
        ref[0, int(single)] += 1
        self.assertTrue((oh_single - ref).pow(2).sum().item() == 0)

        single = np.float64(3.2)
        oh_single = ch.onehot(single, dim=DIM)
        self.assertTrue(oh_single.size(0) == 1)
        self.assertTrue(oh_single.size(1) == DIM)
        ref = th.zeros(1, DIM)
        ref[0, int(single)] += 1
        self.assertTrue((oh_single - ref).pow(2).sum().item() == 0)

        multi = th.arange(DIM)
        multi = ch.onehot(multi, dim=DIM)
        ref = th.eye(DIM)
        self.assertTrue((multi - ref).pow(2).sum().item() == 0)


if __name__ == '__main__':
    unittest.main()
