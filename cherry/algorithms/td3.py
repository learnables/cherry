# -*- coding=utf-8 -*-

import dataclasses
import torch
import cherry

from .arguments import AlgorithmArguments


@dataclasses.dataclass
class TD3(AlgorithmArguments):

    """
    <a href="https://github.com/learnables/cherry/blob/master/cherry/algorithms/td3.py" class="source-link">[Source]</a>

    ## Description

    Utilities to implement TD3 from [1].

    The main idea behind TD3 is to extend DDPG with *twin* action value functions.
    Namely, the action values are computed with:

    $$
    \\min_{i=1, 2} Q_i(s_t, \\pi(s_t) + \\epsilon),
    $$

    where \(\\pi\) is a deterministic policy and \(\\epsilon\) is (typically) sampled from a Gaussian distribution.
    See [cherry.nn.Twin](/api/cherry.nn/#cherry.nn.action_value.Twin) to easily implement such twin Q-functions.

    The authors also suggest to delay the updates to the policy.
    This simply boils down to applying 1 policy update every N times the action value function is updated.
    This implementation also supports delaying updates to the action value and its target network.

    ## References

    1. Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods", ICML 2018.

    ## Arguments

    * `batch_size` (int, *optional*, default=512) - Number of samples to get from the replay.
    * `discount` (float, *optional*, default=0.99) - Discount factor.
    * `policy_delay` (int, *optional*, default=1) - Delay between policy updates.
    * `target_delay` (int, *optional*, default=1) - Delay between action value updates.
    * `target_polyak_weight` (float, *optional*, default=0.995) - Weight factor `alpha` for Polyak averaging;
        see [cherry.models.polyak_average](/api/cherry.models/#cherry.models.utils.polyak_average).
    * `nsteps` (int, *optional*, default=1) - Number of bootstrapping steps to compute the target values.

    """

    batch_size: int = 512
    discount: float = 0.99
    policy_delay: int = 1
    target_delay: int = 1
    target_polyak_weight: float = 0.995
    nsteps: int = 1

    def update(
        self,
        replay,
        policy,
        action_value,
        target_action_value,
        policy_optimizer,
        action_value_optimizer,
        update_policy=True,
        update_target=True,
        update_value=True,
        device=None,
        **kwargs,
    ):

        """
        ## Description

        Implements a single TD3 update.

        ## Arguments

        * `replay` (cherry.ExperienceReplay) - Offline replay to sample transitions from.
        * `policy` (cherry.nn.Policy) - Policy to optimize.
        * `action_value` (cherry.nn.ActionValue) - Twin action value to optimize; see cherry.nn.Twin.
        * `target_action_value` (cherry.nn.ActionValue) - Target action value.
        * `policy_optimizer` (torch.optim.Optimizer) - Optimizer for the `policy`.
        * `action_value_optimizer` (torch.optim.Optimizer) - Optimizer for the `action_value`.
        * `update_policy` (bool, *optional*, default=True) - Whether to update the policy.
        * `update_target` (bool, *optional*, default=True) - Whether to update the action value target network.
        * `update_value` (bool, *optional*, default=True) - Whether to update the action value.
        * `device` (torch.device) - The device used to compute the update.

        """
        # Log debugging values
        stats = {}

        # unwrap hyper-parameters
        config = self.unpack_config(self, kwargs)

        # fetch batch
        batch = replay.sample(
            config.batch_size,
            nsteps=config.nsteps,
            discount=config.discount,
        )
        states = batch.state().to(device, non_blocking=True).float()
        next_states = batch.next_state().to(device, non_blocking=True).float()
        actions = batch.action().to(device, non_blocking=True)
        rewards = batch.reward().to(device, non_blocking=True)
        dones = batch.done().to(device, non_blocking=True)

        # Update Policy
        if update_policy:
            new_actions = policy(states.detach()).rsample()
            policy_loss = - action_value(states.detach(), new_actions).mean()

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()
            stats['td3/policy_loss'] = policy_loss.item()

        # Update Q-function
        if update_value:
            qf1_estimate, qf2_estimate = action_value.twin(
                states,
                actions.detach(),
            )

            # compute targets
            with torch.no_grad():
                next_actions = policy(next_states).sample()
                target_q = target_action_value(next_states, next_actions)

            target_q = rewards + (1. - dones) * config.discount * target_q
            critic_qf1_loss = (qf1_estimate - target_q).pow(2).mean().clamp(-1, 5e4)
            critic_qf2_loss = (qf2_estimate - target_q).pow(2).mean().clamp(-1, 5e4)
            value_loss = (critic_qf1_loss + critic_qf2_loss) / 2.0

            # Update Critic Networks
            action_value_optimizer.zero_grad()
            value_loss.backward()
            action_value_optimizer.step()

            stats['td3/qf_loss1'] = critic_qf1_loss.item()
            stats['td3/qf_loss2'] = critic_qf2_loss.item()
            stats['td3/batch_rewards'] = rewards.mean().item()

        # Move target approximator parameters towards critic
        if update_target:
            cherry.models.polyak_average(
                source=target_action_value,
                target=action_value,
                alpha=config.target_polyak_weight,
            )

        return stats
