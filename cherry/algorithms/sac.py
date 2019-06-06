#!/usr/bin/env python3

"""
**Description**

Helper functions for implementing SAC.
"""

import torch as th
from torch.nn import functional as F


def policy_loss(log_probs, q_curr, alpha=1.0):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/sac.py)

    **Description**

    The policy loss of the Soft Actor-Critic.

    New actions are sampled from the target policy, and those are used to compute the Q-values.
    While we should back-propagate through the Q-values to the policy parameters, we shouldn't
    use that gradient to optimize the Q parameters.
    This is often avoided by either using a target Q function, or by zero-ing out the gradients
    of the Q function parameters.

    **References**

    1. Haarnoja et al. 2018. “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.” arXiv [cs.LG].
    2. Haarnoja et al. 2018. “Soft Actor-Critic Algorithms and Applications.” arXiv [cs.LG].

    **Arguments**

    * **log_probs** (tensor) - Log-density of the selected actions.
    * **q_curr** (tensor) - Q-values of state-action pairs.
    * **alpha** (float, *optional*, default=1.0) - Entropy weight.

    **Returns**

    * (tensor) - The policy loss for the given arguments.

    **Example**

    ~~~python
    densities = policy(batch.state())
    actions = densities.sample()
    log_probs = densities.log_prob(actions)
    q_curr = q_function(batch.state(), actions)
    loss = policy_loss(log_probs, q_curr, alpha=0.1)
    ~~~

    """
    msg = 'log_probs and q_curr must have equal size.'
    assert log_probs.size() == q_curr.size(), msg
    return th.mean(alpha * log_probs - q_curr)


def action_value_loss(q_old, v_next, rewards, dones, gamma):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/sac.py)

    **Description**

    The action-value loss of the Soft Actor-Critic.

    For a given transition, the Q-values and V-values are recomputed on the actual transition data.
    The we should not back-propagate through the V-values, and often use a target network for them.

    **References**

    1. Haarnoja et al. 2018. “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.” arXiv [cs.LG].
    2. Haarnoja et al. 2018. “Soft Actor-Critic Algorithms and Applications.” arXiv [cs.LG].

    **Arguments**

    * **q_old** (tensor) - Action values of the actual transition.
    * **v_next** (tensor) - State values of the resulting state.
    * **rewards** (tensor) - Observed rewards of the transition.
    * **dones** (tensor) - Which states were terminal.
    * **gamma** (float) - Discount factor.

    **Returns**

    * (tensor) - The policy loss for the given arguments.

    **Example**

    ~~~python
    q_old = qf(batch.state(), batch.action().detach())
    v_next = targe_vf(batch.next_state())
    loss = action_value_loss(q_old,
                             v_next,
                             batch.reward(),
                             batch.done(),
                             gamma=0.99)
    ~~~

    """
    msg = 'v_next, rewards, and dones must have equal size.'
    assert rewards.size() == dones.size() == v_next.size(), msg
    q_target = rewards + (1.0 - dones) * gamma * v_next
    return F.mse_loss(q_old, q_target.detach())


def state_value_loss(v_curr, log_probs, q_curr, alpha=1.0):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/sac.py)

    **Description**

    The state-value loss of the Soft Actor-Critic.

    This update is computed "on-policy": states are sampled from a replay but the state values,
    action values, and log-densities are computed using the current value functions and policy.

    **References**

    1. Haarnoja et al. 2018. “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.” arXiv [cs.LG].
    2. Haarnoja et al. 2018. “Soft Actor-Critic Algorithms and Applications.” arXiv [cs.LG].

    **Arguments**

    * **v_curr** (tensor) - State values for some observed states.
    * **log_probs** (tensor) - Log-density of actions sampled from the current policy.
    * **q_curr** (tensor) - Action values of the actions for the current policy.
    * **alpha** (float, *optional*, default=1.0) - Entropy weight.

    **Returns**

    * (tensor) - The state value loss for the given arguments.

    **Example**

    ~~~python
    densities = policy(batch.state())
    actions = densities.sample()
    log_probs = densities.log_prob(actions)
    q_curr = qf(batch.state(), actions)
    v_curr = vf(batch.state())
    loss = state_value_loss(v_curr,
                            log_probs,
                            q_curr,
                            alpha=0.1)
    ~~~

    """
    msg = 'v_curr, q_curr, and log_probs must have equal size.'
    assert v_curr.size() == q_curr.size() == log_probs.size(), msg
    v_target = q_curr - alpha * log_probs
    return F.mse_loss(v_curr, v_target.detach())


def entropy_weight_loss(log_alpha, log_probs, target_entropy):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/sac.py)

    **Description**

    Loss of the entropy weight, to automatically tune it.

    The target entropy needs to be manually tuned.
    However, a popular heuristic for TanhNormal policies is to use the negative of the action-space
    dimensionality. (e.g. -4 when operating the voltage of a quad-rotor.)

    **References**

    1. Haarnoja et al. 2018. “Soft Actor-Critic Algorithms and Applications.” arXiv [cs.LG].

    **Arguments**

    * **log_alpha** (tensor) - Log of the entropy weight.
    * **log_probs** (tensor) - Log-density of policy actions.
    * **target_entropy** (float) - Target of the entropy value.

    **Returns**

    * (tensor) - The state value loss for the given arguments.

    **Example**

    ~~~python
    densities = policy(batch.state())
    actions = densities.sample()
    log_probs = densities.log_prob(actions)
    target_entropy = -np.prod(env.action_space.shape).item()
    loss = entropy_weight_loss(alpha.log(),
                               log_probs,
                               target_entropy)
    ~~~

    """
    loss = -(log_alpha * (log_probs + target_entropy).detach())
    return loss.mean()
