#!/usr/bin/env python3

import torch as th
from torch.nn import functional as F


def policy_loss(log_probs, q_actions, alpha=1.0):
    """
    Arguments:

    * log_probs: the log density of the actions from the current policy on
               some states.
    * q_actions: the Q-values for those same actions.
    * alpha: the weight of the policy entropy.
    """
    msg = 'log_probs and q_actions must have equal size.'
    assert log_probs.size() == q_actions.size(), msg
    return th.mean(alpha * log_probs - q_actions)


def action_value_loss(q_old_pred, v_next, rewards, dones, gamma):
    """
    Arguments:

    * q_old_pred: Q values on an existing transition.
    * v_next: V values for the resulting state.
    * rewards: observed rewards during the transition.
    * dones: which states were terminal.
    * gamma: discount factor.
    """
    msg = 'v_next, rewards, and dones must have equal size.'
    assert rewards.size() == dones.size() == v_next.size(), msg
    q_target = rewards + (1.0 - dones) * gamma * v_next
    return F.mse_loss(q_old_pred, q_target.detach())


def state_value_loss(v_pred, log_probs, q_values, alpha=1.0):
    """
    Arguments:

    * v_pred: the V values of states from a batch.
    * log_probs: the log density of actions from the current policy on
               those states.
    * q_values: the Q values of the those actions on those states.
    * alpha: the weight of the policy entropy.
    """
    msg = 'v_pred, q_values, and log_probs must have equal size.'
    assert v_pred.size() == q_values.size() == log_probs.size(), msg
    v_target = q_values - alpha * log_probs
    return F.mse_loss(v_pred, v_target.detach())


def entropy_weight_loss(log_alpha, log_probs, target_entropy):
    loss = -(log_alpha * (log_probs + target_entropy).detach())
    return loss.mean()
