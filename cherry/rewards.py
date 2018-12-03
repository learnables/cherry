#!/usr/bin/env python3


def discount_rewards(gamma, rewards, dones):
    """
    TODO: Assuming we're given a partial trajectory,
    I think that the first R should be bootstraped from the value function.
    Otherwise, you might have an all 0 discounted rewards.
    (IIRC, that's how they do it in A2C)
    But is it here the best place to take care of this ?
    """
    R = 0.0
    discounted = []
    for reward, done in zip(rewards[::-1], dones[::-1]):
        if done:
            R = 0.0
        R = reward + gamma * R
        discounted.insert(0, R)
    return discounted
