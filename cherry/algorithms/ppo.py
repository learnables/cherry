#!/usr/bin/env python3

import dataclasses
import torch
import cherry
import dotmap

from cherry import debug
from .arguments import AlgorithmArguments


@dataclasses.dataclass
class PPO(AlgorithmArguments):

    """
    <a href="https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/ppo.py" class="source-link">[Source]</a>

    ## Description

    Utilities to implement PPO from [1].

    The idea behing PPO is to cheaply approximate TRPO's trust-region with the following objective:

    $$
    \\mathbb{E}[\\min(
        \\frac{\\pi_{\\theta}}{\\pi_{\\textrm{old}}} \\cdot A,
        \\mathrm{clip}(\\frac{\\pi_{\\theta}}{\\pi_{\\textrm{old}}} \\cdot A, 1-\\epsilon, 1+\\epsilon)
    )],
    $$

    where \(\\pi_{\\theta}\) is the current policy and \(\\pi_{\\mathrm{old}}\) is the policy used to collect the online replay's data.

    ## References

    1. Schulman et al., “Proximal Policy Optimization Algorithms”, ArXiv 2017.

    ## Arguments

    Note: the following arguments were optimized for continuous control on PyBullet / MuJoCo.

    * `num_steps` (int, *optional*, default=320) - Number of of PPO gradient steps in a single update.
    * `batch_size` (int, *optional*, default=512) - Number of samples to get from the replay.
    * `policy_clip` (float, *optional*, default=0.2) - Clip constant for the policy.
    * `value_clip` (float, *optional*, default=0.2) - Clip constant for state value function.
    * `value_weight` (float, *optional*, default=0.5) - Scaling factor fo the state value function penalty.
    * `entropy_weight` (float, *optional*, default=0.0) - Scaling factor of the entropy penalty.
    * `discount` (float, *optional*, default=0.99) - Discount factor.
    * `gae_tau` (float, *optional*, default=0.95) - Bias-variance trade-off for the generalized advantage estimator.
    * `gradient_norm` (float, *optional*, default=0.5) - Maximum gradient norm.
    * `eps` (float, *optional*, default=0.5) - Numerical stability constant.
    """

    num_steps: int = 320
    batch_size: float = 64
    policy_clip: float = 0.2
    value_clip: float = 0.2
    value_weight: float = 0.5
    entropy_weight: float = 0.0
    discount: float = 0.99
    gae_tau: float = 0.95
    gradient_norm: float = 0.5  # no clipping if 0
    eps: float = 1e-8

    @staticmethod
    def policy_loss(new_log_probs, old_log_probs, advantages, clip=0.1):
        """
        ## Description

        The clipped policy loss of Proximal Policy Optimization.

        ## Arguments

        * `new_log_probs` (tensor) - The log-density of actions from the target policy.
        * `old_log_probs` (tensor) - The log-density of actions from the behaviour policy.
        * `advantages` (tensor) - Advantage of the actions.
        * `clip` (float, *optional*, default=0.1) - The clipping coefficient.

        ## Returns

        * loss (tensor) - The clipped policy loss for the given arguments.

        **Example**

        ~~~python
        advantage = ch.pg.generalized_advantage(
            GAMMA,
            TAU,
            replay.reward(),
            replay.done(),
            replay.value(),
            next_state_value,
        )
        new_densities = policy(replay.state())
        new_logprobs = new_densities.log_prob(replay.action())
        loss = policy_loss(
            new_logprobs,
            replay.logprob().detach(),
            advantage.detach(),
            clip=0.2,
        )
        ~~~
        """
        msg = 'new_log_probs, old_log_probs and advantages must have equal size.'
        assert new_log_probs.size() == old_log_probs.size() == advantages.size(),\
            msg
        if debug.IS_DEBUGGING:
            if old_log_probs.requires_grad:
                debug.logger.warning('PPO:policy_loss: old_log_probs.requires_grad is True.')
            if advantages.requires_grad:
                debug.logger.warning('PPO:policy_loss: advantages.requires_grad is True.')
            if not new_log_probs.requires_grad:
                debug.logger.warning('PPO:policy_loss: new_log_probs.requires_grad is False.')
        ratios = torch.exp(new_log_probs - old_log_probs)
        obj = ratios * advantages
        obj_clip = ratios.clamp(1.0 - clip, 1.0 + clip) * advantages
        return - torch.min(obj, obj_clip).mean()

    @staticmethod
    def state_value_loss(new_values, old_values, rewards, clip=0.1):
        """
        ## Description

        The clipped state-value loss of Proximal Policy Optimization.

        ## Arguments

        * `new_values` (tensor) - State values from the optimized value function.
        * `old_values` (tensor) - State values from the reference value function.
        * `rewards` (tensor) -  Observed rewards.
        * `clip` (float, *optional*, default=0.1) - The clipping coefficient.

        ## Returns

        * loss (tensor) - The clipped value loss for the given arguments.

        ## Example

        ~~~python
        values = v_function(batch.state())
        value_loss = ppo.state_value_loss(
            values,
            batch.value().detach(),
            batch.reward(),
            clip=0.2,
        )
        ~~~
        """
        msg = 'new_values, old_values, and rewards must have equal size.'
        assert new_values.size() == old_values.size() == rewards.size(), msg
        if debug.IS_DEBUGGING:
            if old_values.requires_grad:
                debug.logger.warning('PPO:state_value_loss: old_values.requires_grad is True.')
            if rewards.requires_grad:
                debug.logger.warning('PPO:state_value_loss: rewards.requires_grad is True.')
            if not new_values.requires_grad:
                debug.logger.warning('PPO:state_value_loss: new_values.requires_grad is False.')
        loss = (rewards - new_values)**2
        clipped_values = old_values + (new_values - old_values).clamp(-clip, clip)
        clipped_loss = (rewards - clipped_values)**2
        return 0.5 * torch.max(loss, clipped_loss).mean()

    @staticmethod
    def _mean(elements):
        length = len(elements)
        if length > 0:
            return sum(elements) / float(length)
        return 0.0

    def update(
        self,
        replay,
        policy,
        optimizer,
        state_value,
        **kwargs,
    ):
        """
        ## Description

        Implements a single PPO update.

        ## Arguments

        * `replay` (cherry.ExperienceReplay) - Online replay to sample transitions from.
        * `policy` (cherry.nn.Policy) - Policy to optimize.
        * `state_value` (cherry.nn.StateValue) - State value function \(V(s)\).
        * `optimizer` (torch.optim.Optimizer) - Optimizer for the `policy`.
        """
        # Log debugging values
        stats = dotmap.DotMap()

        # Unpack arguments and variables
        config = self.unpack_config(self, kwargs)
        vectorized = replay.vectorized

        # Process replay
        all_states = replay.state()
        all_actions = replay.action()
        all_dones = replay.done()
        all_rewards = replay.reward()
        with torch.no_grad():
            if vectorized:
                state_shape = all_states.shape[2:]
                action_shape = all_actions.shape[2:]
                all_log_probs = policy.log_prob(
                    all_states.reshape(-1, *state_shape),
                    all_actions.reshape(-1, *action_shape)
                )
                # reshape to -1 here because maybe Normal distribution.
                all_log_probs = all_log_probs.reshape(*all_states.shape[:2], -1)
                all_values = state_value(all_states.reshape(-1, *state_shape))
                all_values = all_values.reshape(*all_states.shape[:2], 1)
            else:
                all_log_probs = policy.log_prob(all_states, all_actions)
                all_values = state_value(all_states)

        # Compute advantages and returns
        next_state_value = state_value(replay[-1].next_state)
        all_advantages = cherry.pg.generalized_advantage(
            config.discount,
            config.gae_tau,
            all_rewards,
            all_dones,
            all_values,
            next_state_value,
        )

        returns = all_advantages + all_values
        all_advantages = cherry.normalize(all_advantages, epsilon=config.eps)

        for i, sars in enumerate(replay):
            sars.log_prob = cherry.totensor(all_log_probs[i].detach())
            sars.value = cherry.totensor(all_values[i].detach())
            sars.advantage = cherry.totensor(all_advantages[i].detach())
            sars.retur = cherry.totensor(returns[i].detach())

        # Logging
        policy_losses = []
        entropies = []
        value_losses = []

        # avoids the weird shapes later in the loop and extra forward passes.
        replay = replay.flatten()

        # Perform some optimization steps
        for step in range(config.num_steps):
            batch = replay.sample(config.batch_size)
            states = batch.state()
            advantages = batch.advantage()

            new_densities = policy(states)
            new_values = state_value(states)

            # Compute losses
            new_log_probs = new_densities.log_prob(batch.action())
            entropy = new_densities.entropy().mean()
            policy_loss = PPO.policy_loss(
                new_log_probs,
                batch.log_prob(),
                advantages,
                clip=config.policy_clip,
            )
            value_loss = PPO.state_value_loss(
                new_values,
                batch.value(),
                batch.retur(),
                clip=config.value_clip,
            )
            loss = policy_loss
            loss = loss + config.value_weight * value_loss
            loss = loss - config.entropy_weight * entropy

            # Take optimization step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                config.gradient_norm,
            )
            optimizer.step()

            policy_losses.append(policy_loss)
            entropies.append(entropy)
            value_losses.append(value_loss)

        # Log metrics
        stats['ppo/policy_loss'] = PPO._mean(policy_losses)
        stats['ppo/entropies'] = PPO._mean(entropies)
        stats['ppo/value_loss'] = PPO._mean(value_losses)
        return stats


policy_loss = PPO.policy_loss
state_value_loss = PPO.state_value_loss
