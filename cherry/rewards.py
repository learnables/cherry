#!/usr/bin/env python3


def discount_rewards(gamma, rewards, dones, bootstrap=0.0):
    """
    rewards: 0.1119, 0.1131, 0.1142, 0.1154, 0.1165
    advantages: 0.1120, 0.1132, 0.1143, 0.1155, 0.1166
    """
    R = bootstrap
    discounted = []
    for reward, done in zip(rewards[::-1], dones[::-1]):
        if done:
            R = 0.0
        R = reward + gamma * R
        discounted.insert(0, R)
    return discounted


def generalized_advantage_estimate(gamma, tau, rewards, dones, values, bootstrap=0.0):
    """
    rewards: 0.1119, 0.1131, 0.1142, 0.1154, 0.1165
    advantages: -0.1806, -0.1602, -0.1278, -0.1558, -0.0566
    """
    msg = 'GAE requires the value of the next state to be appended.'
    assert len(values) > len(rewards), msg
    R = bootstrap
    advantage = 0.0
    discounted = []
    advantages = []
    for i in reversed(range(len(rewards))):
        if dones[i]:
            R = 0.0
            advantage = 0.0
        R = rewards[i] + gamma * R
        discounted.insert(0, R)
        td_error = rewards[i] + gamma * values[i+1] - values[i]
        advantage = advantage * tau * gamma + td_error
        advantages.insert(0, advantage)

    return discounted, advantages


discount = discount_rewards
gae = generalized_advantage_estimate
