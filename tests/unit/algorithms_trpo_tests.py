#!/usr/bin/env python3

import unittest
import torch as th

from cherry.algorithms import trpo

H_SIZE = 10
NUM_TRIALS = 10
BSZ = 10


def close(a, b, eps=1e-5):
    return (a - b).norm(p=2) <= eps


class TestTRPOAlgorithms(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_trpo_policy_loss(self):

        for shape in [(1, BSZ), (BSZ, 1), (BSZ, )]:
            new_log_probs = th.randn(BSZ)
            old_log_probs = th.randn(BSZ)
            advantages = th.randn(BSZ)
            ref = trpo.policy_loss(new_log_probs, old_log_probs, advantages)
            loss = trpo.policy_loss(new_log_probs.view(*shape),
                                    old_log_probs.view(*shape),
                                    advantages.view(*shape))
            self.assertAlmostEqual(loss.item(), ref.item())


    def test_trpo_hvp(self):
        for _ in range(NUM_TRIALS):
            H = th.randn(H_SIZE, H_SIZE)
            H = H @ H.t()
            p = th.randn(H_SIZE, requires_grad=True)
            v = th.randn(H_SIZE)
            loss = 0.5 * p.t() @ H @ p
            hvp = trpo.hessian_vector_product(loss, p, damping=0.0)
            Hv = H @ v
            hvp = hvp(v)
            self.assertTrue(close(Hv, hvp))

    def test_trpo_cg(self):
        th.manual_seed(42)
        for _ in range(NUM_TRIALS):
            H = th.randn(H_SIZE, H_SIZE)
            H = H @ H.t()
            p = th.randn(H_SIZE, requires_grad=True)
            v = th.randn(H_SIZE)
            loss = 0.5 * p.t() @ H @ p
            hvp = trpo.hessian_vector_product(loss, p, damping=0.0)
            Hiv = H.pinverse() @ v
            x = trpo.conjugate_gradient(hvp, v, num_iterations=1000)
            self.assertTrue(close(Hiv, x, eps=1e-1))


if __name__ == '__main__':
    unittest.main()
