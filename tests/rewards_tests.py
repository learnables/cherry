import unittest
import random
import numpy as np
import torch as th
import cherry as ch
from cherry.rewards import discount_rewards
from cherry.utils import normalize
import copy

GAMMA = 0.5
NUM_SAMPLES = 10
VECTOR_SIZE = 5

class TestRewards(unittest.TestCase):
    def setUp(self):
        self.replay = ch.ExperienceReplay()

    def tearDown(self):
        pass

    def test_discount(self):
        vector = np.random.rand(VECTOR_SIZE)
        for i in range(5):
            self.replay.append(vector,
                               vector,
                               8,
                               vector,
                               False)
        self.replay.storage['dones'][-1] += 1
        discounted = discount_rewards(GAMMA,
                                      self.replay.rewards,
                                      self.replay.dones,
                                      bootstrap=0)
        print(discounted)

#    def test_discount_rewards(self):
#        standard_replay = self.replay
#        vector = np.random.rand(VECTOR_SIZE)
#        for i in range(NUM_SAMPLES):
#            standard_replay.add(vector,
#                            vector,
#                            i,
#                            vector,
#                            False,
#                            info={'vector': vector})
#
#        standard_replay.storage['dones'][9] += 1
#        print (standard_replay.rewards)
#
#        # get 4 rounds of rewards, but we are only dealing with 2nd and 3rd
#        first_rewards = discount_rewards(GAMMA, standard_replay.rewards, standard_replay.dones)
#        second_rewards = discount_rewards(GAMMA, first_rewards, standard_replay.dones)
#        third_rewards = discount_rewards(GAMMA, second_rewards, standard_replay.dones)
#        fourth_rewards = discount_rewards(GAMMA, third_rewards, standard_replay.dones)
#
#        # 4th to the 10th element for rewards => 6 elements in total
#        second_splice_reward_part = copy.deepcopy(second_rewards);
#        second_splice_reward_part = second_splice_reward_part[4: 10]
#        # 4th to the 10th element for dones => 6 elements in total
#        second_splice_done_part = copy.deepcopy(standard_replay.dones);
#        second_splice_done_part = second_splice_done_part[4: 10]
#        # print ('\n', 'SECOND DONE SPLICE: ', second_splice_done_part)
#
#        # 0th to the 4th element for rewards => 4 elements in total
#        third_splice_reward_part = copy.deepcopy(third_rewards);
#        third_splice_reward_part = third_splice_reward_part[0: 4]
#        # 0th to the 4th element for rewards => 4 elements in total
#        third_splice_done_part = copy.deepcopy(standard_replay.dones);
#        third_splice_done_part = second_splice_done_part[0: 4]
#        # print ('\n', 'THIRD DONE SPLICE', third_splice_done_part)
#
#        second_third_reward_splice = second_splice_reward_part + third_splice_reward_part
#
#        second_third_done_splice = th.cat((second_splice_done_part, third_splice_done_part), dim=0)
#        # print ('dsahuifewgio uiquhif', second_splice_done_part)
#
#        # print ('STRNDA', second_third_done_splice)
#
#        print ('\n')
#        print ('FIRST IS: ', first_rewards, '\n')
#        print ('SECOND IS: ', second_rewards, '\n')
#        print ('THIRD IS: ', third_rewards, '\n')
#
#        test_rewards = discount_rewards(GAMMA,
#                                        second_third_reward_splice,
#                                        second_third_done_splice,
#                                        bootstrap=third_rewards[9])
#
#        print ('SPLICED CALCULATED IS: ', test_rewards, '\n')
#        print (  (second_rewards + third_rewards)[4:14] )



if __name__ == '__main__':
    unittest.main()
