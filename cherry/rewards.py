#!/usr/bin/env python3


def discount_rewards(gamma, rewards, dones, bootstrap=0.0):
    """
    rewards: 0.1119, 0.1131, 0.1142, 0.1154, 0.1165
    advantages: 0.1120, 0.1132, 0.1143, 0.1155, 0.1166
    """
    R = bootstrap
    discounted = []
    length = len(rewards)
    for t in reversed(range(length)):
        if dones[t]:
            R = 0.0
        R = rewards[t] + gamma * R
        discounted.insert(0, R)
    return discounted


def generalized_advantage_estimate(gamma, tau, rewards, dones, values):
    """
    rewards: 0.1119, 0.1131, 0.1142, 0.1154, 0.1165
    advantages: -0.1806, -0.1602, -0.1278, -0.1558, -0.0566
    """
    msg = 'GAE requires the value of the next state to be appended.'
    assert len(values) > len(rewards), msg
    advantages = []
    advantage = 0
    for i in reversed(range(len(rewards))):
        td_error = rewards[i] + (1.0 - dones[i]) * gamma * values[i+1] - values[i]
        advantage = advantage * tau * gamma * (1.0 - dones[i]) + td_error
        advantages.insert(0, advantage)

    return advantages


discount = discount_rewards
gae = generalized_advantage_estimate
