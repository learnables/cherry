#!/usr/bin/env python3


def discount_rewards(gamma, rewards, dones, bootstrap=0.0):
    R = bootstrap
    discounted = []
    length = len(rewards)
    for t in reversed(range(length)):
        if dones[t]:
            R *= 0.0
        R = rewards[t] + gamma * R
        discounted.insert(0, R)
    return discounted


def generalized_advantage_estimate(gamma, tau, rewards, dones, values, next_value):
    msg = 'GAE needs as many rewards, values and dones.'
    assert len(values) == len(rewards) == len(dones), msg
    advantages = []
    advantage = 0
    for i in reversed(range(len(rewards))):
        td_error = rewards[i] + (1.0 - dones[i]) * gamma * next_value - values[i]
        advantage = advantage * tau * gamma * (1.0 - dones[i]) + td_error
        advantages.insert(0, advantage)
        next_value = values[i]
    return advantages


discount = discount_rewards
gae = generalized_advantage_estimate
