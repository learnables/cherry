#!/usr/bin/env python3

import unittest


def train_cherry():
    result = {
        'rewards',
        'policy_losses',
        'value_losses',
        'weights',
    }
    return result


def train_spinup():
    result = {
        'rewards',
        'policy_losses',
        'value_losses',
        'weights',
    }
    return result


def close(a, b):
    return (a-b).norm(p=2) <= 1e-8


class TestSpinningUpPPO(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_vpg(self):
        cherry = train_cherry()
        spinup = train_spinup()

        # Check rewards
        for rc, rs in zip(cherry['rewards'], spinup['rewards']):
            self.assertEqual(rc, rs)

        # Check policy loss
        for pc, ps in zip(cherry['policy_losses'], spinup['policy_losses']):
            self.assertEqual(pc, ps)

        # Check value loss
        for vc, vs in zip(cherry['value_losses'], spinup['value_losses']):
            self.assertEqual(vc, vs)

        # Check weights
        for wc, ws in zip(cherry['weights'], spinup['weights']):
            self.assertTrue(close(wc, ws))


if __name__ == "__main__":
    unittest.main()
