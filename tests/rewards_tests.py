import unittest
import torch as th
import cherry as ch
from cherry.rewards import discount, generalized_advantage

GAMMA = 0.5
NUM_SAMPLES = 10
VECTOR_SIZE = 5


def close(a, b):
    return (a - b).norm(p=2) <= 1e-8


def discount_rewards(gamma, rewards, dones, bootstrap=0.0):
    """
    Implementation that works with lists.
    """
    R = bootstrap
    discounted = []
    length = len(rewards)
    for t in reversed(range(length)):
        if dones[t]:
            R *= 0.0
        R = rewards[t] + gamma * R
        discounted.insert(0, R)
    return discounted


class TestRewards(unittest.TestCase):
    def setUp(self):
        self.replay = ch.ExperienceReplay()

    def tearDown(self):
        pass

    def test_discount(self):
        vector = th.randn(VECTOR_SIZE)
        for i in range(5):
            self.replay.append(vector,
                               vector,
                               8,
                               vector,
                               False)
        self.replay.storage['dones'][-1] += 1
        discounted = discount(GAMMA,
                              self.replay.rewards,
                              self.replay.dones,
                              bootstrap=0)
        ref = th.Tensor([15.5, 15.0, 14.0, 12.0, 8.0]).view(-1, 1)
        self.assertTrue(close(discounted, ref))

        # Test overlapping episodes with bootstrap
        overlap = self.replay[2:] + self.replay[:3]
        overlap_discounted = discount(GAMMA,
                                      overlap.rewards,
                                      overlap.dones,
                                      bootstrap=discounted[3])
        ref = th.cat((discounted[2:], discounted[:3]), dim=0)
        self.assertTrue(close(overlap_discounted, ref))

    def test_generalized_advantage(self):
        pass


if __name__ == '__main__':
    unittest.main()
