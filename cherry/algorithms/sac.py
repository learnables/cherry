#!/usr/bin/env python3

import torch as th
from torch.nn import functional as F


def policy_loss(log_probs, q_actions, alpha=1.0):
    """
    **Description**

    To calculate the policy loss in SAC method.

    **Arguments**

    * **log_probs** - The log density of the actions from the current policy on some states.
    * **q_actions** - The Q-values for those same actions.
    * **alpha** - The weight of the policy entropy.

    **Reference**

    **Example**
    ~~~python
    log_probs = density.rsample_and_log_prob()
    log_probs = log_probs.sum(dim=1, keepdim=True)
    q_values = qf(batch.state(), actions)
    policy_loss = sac.policy_loss(log_probs, q_values, alpha)
    ~~~    


    """
    msg = 'log_probs and q_actions must have equal size.'
    assert log_probs.size() == q_actions.size(), msg
    return th.mean(alpha * log_probs - q_actions)


def action_value_loss(q_old_pred, v_next, rewards, dones, gamma):
    """
    **Description**

    To calculate the action value loss in SAC method.

    **Arguments**

    * **q_old_pred** - Q values on an existing transition.
    * **v_next** - V values for the resulting state.
    * **rewards** - Observed rewards during the transition.
    * **dones** - Which states were terminal.
    * **gamma** - Discount factor.

    **Reference**

    **Example**
    ~~~python
    q_old_pred = qf(batch.state(), batch.action().detach())
    v_next = target_vf(batch.next_state())

    qf_loss = sac.action_value_loss(q_old_pred,
                                    v_next,
                                    batch.reward(),
                                    batch.done(),
                                    GAMMA=0.99)
    ~~~

    """

    msg = 'v_next, rewards, and dones must have equal size.'
    assert rewards.size() == dones.size() == v_next.size(), msg
    q_target = rewards + (1.0 - dones) * gamma * v_next
    return F.mse_loss(q_old_pred, q_target.detach())


def state_value_loss(v_pred, log_probs, q_values, alpha=1.0):
    """
    **Description**

    To calculate the action value loss in SAC method.

    **Arguments**

    * **v_pred** - The V values of states from a batch.
    * **log_probs** - The log density of actions from the current policy on those states.
    * **q_values** - The Q values of  those actions on those states.
    * **alpha** - The weight of the policy entropy.

    **Reference**

    **Example**
    ~~~python
    v_pred = vf(batch.state())
    log_probs = density.rsample_and_log_prob()
    log_probs = log_probs.sum(dim=1, keepdim=True)
    q_values = qf(batch.state(), actions)
    vf_loss = sac.state_value_loss(v_pred, log_probs, q_values, alpha)
    ~~~    

    """
    msg = 'v_pred, q_values, and log_probs must have equal size.'
    assert v_pred.size() == q_values.size() == log_probs.size(), msg
    v_target = q_values - alpha * log_probs
    return F.mse_loss(v_pred, v_target.detach())


def entropy_weight_loss(log_alpha, log_probs, target_entropy):
    """
    **Description**

    To calculate the entropy weight loss.

    **Arguments**

    * **log_alpha** - The log of the entropy penalty
    * **log_probs** - The log density of actions from the current policy on those states.
    * **target_entropy** - The policy entropy

    **Reference**

    **Example**
    log_alpha = th.zeros(1, requires_grad=True)
    log_probs = density.rsample_and_log_prob()
    log_probs = log_probs.sum(dim=1, keepdim=True)
    target_entropy = -6
    alpha_loss = sac.entropy_weight_loss(log_alpha,
                                         log_probs,
                                         target_entropy)

    """


    loss = -(log_alpha * (log_probs + target_entropy).detach())
    return loss.mean()
