#!/usr/bin/env python3

"""
**Description**

Helper functions for implementing Soft-Actor Critic.

You should update the function approximators according to the following order.

1. Entropy weight update.
2. Action-value update.
3. State-value update. (Optional, c.f. below)
4. Policy update.

Note that most recent implementations of SAC omit step 3. above by using
the Bellman residual instead of modelling a state-value function.
For an example of such implementation refer to
[this link](https://github.com/seba-1511/cherry/blob/master/examples/pybullet/delayed_tsac_pybullet.py).

"""

import dataclasses
import torch
import cherry
import dotmap

from cherry import debug
from .arguments import AlgorithmArguments


@dataclasses.dataclass
class SAC(AlgorithmArguments):

    """
    <a href="" class="source-link">[Source]</a>

    ## Description
    ## Arguments

    * `batch_size` (int) - The number of samples to get from the replay.

    ## Example
    ~~~python
    ~~~
    """

    batch_size: int = 512
    discount: float = 0.99
    use_automatic_entropy_tuning: bool = True
    policy_delay: int = 2
    target_delay: int = 2
    target_polyak_weight: float = 0.01

    @staticmethod
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
        return torch.mean(alpha * log_probs - q_curr)

    @staticmethod
    def action_value_loss(value, next_value, rewards, dones, gamma):
        """
        [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/sac.py)

        **Description**

        The action-value loss of the Soft Actor-Critic.

        `value` should be the value of the current state-action pair, estimated via the Q-function.
        `next_value` is the expected value of the next state; it can be estimated via a V-function,
        or alternatively by computing the Q-value of the next observed state-action pair.
        In the latter case, make sure that the action is sampled according to the current policy,
        not the one used to gather the data.

        **References**

        1. Haarnoja et al. 2018. “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.” arXiv [cs.LG].
        2. Haarnoja et al. 2018. “Soft Actor-Critic Algorithms and Applications.” arXiv [cs.LG].

        **Arguments**

        * **value** (tensor) - Action values of the actual transition.
        * **next_value** (tensor) - State values of the resulting state.
        * **rewards** (tensor) - Observed rewards of the transition.
        * **dones** (tensor) - Which states were terminal.
        * **gamma** (float) - Discount factor.

        **Returns**

        * (tensor) - The policy loss for the given arguments.

        **Example**

        ~~~python
        value = qf(batch.state(), batch.action().detach())
        next_value = targe_vf(batch.next_state())
        loss = action_value_loss(value,
                                 next_value,
                                 batch.reward(),
                                 batch.done(),
                                 gamma=0.99)
        ~~~

        """
        msg = 'next_value, rewards, and dones must have equal size.'
        assert rewards.size() == dones.size() == next_value.size(), msg
        if debug.IS_DEBUGGING:
            if rewards.requires_grad:
                debug.logger.warning('SAC:action_value_loss: rewards.requires_grad is True.')
            if next_value.requires_grad:
                debug.logger.warning('SAC:action_value_loss: next_value.requires_grad is True.')
            if not value.requires_grad:
                debug.logger.warning('SAC:action_value_loss: value.requires_grad is False.')
        q_target = rewards + (1.0 - dones) * gamma * next_value
        return torch.nn.functional.mse_loss(value, q_target)

    @staticmethod
    def state_value_loss(v_value, log_probs, q_value, alpha=1.0):
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

        * **v_value** (tensor) - State values for some observed states.
        * **log_probs** (tensor) - Log-density of actions sampled from the current policy.
        * **q_value** (tensor) - Action values of the actions for the current policy.
        * **alpha** (float, *optional*, default=1.0) - Entropy weight.

        **Returns**

        * (tensor) - The state value loss for the given arguments.

        **Example**

        ~~~python
        densities = policy(batch.state())
        actions = densities.sample()
        log_probs = densities.log_prob(actions)
        q_value = qf(batch.state(), actions)
        v_value = vf(batch.state())
        loss = state_value_loss(v_value,
                                log_probs,
                                q_value,
                                alpha=0.1)
        ~~~

        """
        msg = 'v_value, q_value, and log_probs must have equal size.'
        assert v_value.size() == q_value.size() == log_probs.size(), msg
        if debug.IS_DEBUGGING:
            if log_probs.requires_grad:
                debug.logger.warning('SAC:state_value_loss: log_probs.requires_grad is True.')
            if q_value.requires_grad:
                debug.logger.warning('SAC:state_value_loss: q_value.requires_grad is True.')
            if not v_value.requires_grad:
                debug.logger.warning('SAC:state_value_loss: v_value.requires_grad is False.')
        v_target = q_value - alpha * log_probs
        return torch.nn.functional.mse_loss(v_value, v_target)

    @staticmethod
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
        if debug.IS_DEBUGGING:
            if log_probs.requires_grad:
                debug.logger.warning('SAC:entropy_weight_loss: log_probs.requires_grad is True.')
            if not log_alpha.requires_grad:
                debug.logger.warning('SAC:entropy_weight_loss: log_alpha.requires_grad is False.')
        loss = -(log_alpha * (log_probs + target_entropy))
        return loss.mean()

    @staticmethod
    def actions_log_probs(density):
        if isinstance(density, cherry.distributions.TanhNormal):
            # NOTE: The following lines are specific to the TanhNormal policy.
            # Other policies should constrain the output of the policy net.
            sampled_actions, log_probs = density.rsample_and_log_prob()
            log_probs = log_probs.sum(dim=-1, keepdim=True)
        else:
            sampled_actions = density.rsample()
            log_probs = density.log_prob(sampled_actions)
            log_probs = log_probs.sum(dim=-1, keepdim=True)
        return sampled_actions, log_probs

    def udpate(
        self,
        replay,
        policy,
        qvalue,
        target_value,
        log_alpha,
        target_entropy,
        policy_optimizer,
        features_optimizer,
        value_optimizer,
        alpha_optimizer,
        features=None,
        target_features=None,
        update_policy=True,
        update_target=False,
        update_value=True,
        update_entropy=True,
        device=None,
        **kwargs,
    ):
        # Log debugging values
        stats = dotmap.DotMap()

        # unpack arguments
        config = self.unpack_config(self, kwargs)

        # Sample mini-batch
        batch = replay.sample(config.batch_size)
        if device is not None:
            batch = batch.to(device, non_blocking=True)
        if batch.vectorized:
            batch = batch.flatten()
        states = batch.state()
        if features is not None:
            states = features(states)

        # Update entropy loss
        if update_entropy:
            density = policy(states.detach())  # detach features per DrQ
            _, log_probs = SAC.actions_log_probs(density)

            if config.use_automatic_entropy_tuning:
                alpha_loss = SAC.entropy_weight_loss(
                    log_alpha,
                    log_probs.detach(),
                    target_entropy,
                )
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()
                alpha = log_alpha.exp()
            else:
                alpha = torch.ones(1)
                alpha_loss = torch.zeros(1)

            stats['sac/alpha_loss'] = alpha_loss.item()
            stats['sac/alpha_value'] = alpha.item()

        # Update Q-function
        if update_value:
            actions = batch.action()
            qf1_estimate, qf2_estimate = qvalue.twin_values(
                states,
                actions.detach(),
            )

            # compute targets
            with torch.no_grad():
                next_states = batch.next_state()
                target_next_states = next_states
                if features is not None:
                    next_states = features(next_states)
                if target_features is not None:
                    target_next_states = target_features(target_next_states)
                next_density = policy(next_states)
                next_actions, next_log_probs = SAC.actions_log_probs(next_density)
                target_q_values = target_value(target_next_states, next_actions) \
                    - alpha * next_log_probs

            rewards = batch.reward().reshape(target_q_values.shape)
            dones = batch.done().reshape(target_q_values.shape)
            critic_qf1_loss = SAC.action_value_loss(
                qf1_estimate,
                target_q_values.detach(),
                rewards,
                dones,
                config.discount,
            )

            critic_qf2_loss = SAC.action_value_loss(
                qf2_estimate,
                target_q_values.detach(),
                rewards,
                dones,
                config.discount,
            )
            value_loss = (critic_qf1_loss + critic_qf2_loss) / 2.0

            # Update Critic Networks
            value_optimizer.zero_grad()
            features_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
            features_optimizer.step()

            stats['sac/qf_loss1'] = critic_qf1_loss.item()
            stats['sac/qf_loss2'] = critic_qf2_loss.item()
            stats['sac/batch_rewards'] = rewards.mean().item()

        # Delayed Updates
        if update_policy:
            # Policy loss
            states = batch.state()  # recompute features
            if features is not None:
                # detach features per DrQ.
                states = features(states).detach()
            density = policy(states)
            new_actions, log_probs = SAC.actions_log_probs(density)
            q_values = qvalue(states, new_actions)
            policy_loss = SAC.policy_loss(log_probs, q_values, alpha.detach())

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            stats['sac/policy_entropy'] = -log_probs.mean()
            stats['sac/policy_loss'] = policy_loss.item()

        # Move target approximator parameters towards critic
        if update_target:
            cherry.models.polyak_average(
                source=target_value,
                target=qvalue,
                alpha=config.target_polyak_weight,
            )
            if features is not None:
                cherry.models.polyak_average(
                    source=target_features,
                    target=features,
                    alpha=config.target_polyak_weight,
                )

        return stats


# legacy
entropy_weight_loss = SAC.entropy_weight_loss
policy_loss = SAC.policy_loss
state_value_loss = SAC.state_value_loss
action_value_loss = SAC.action_value_loss
