# -*- coding=utf-8 -*-

import torch
import cherry
import dataclasses

from .arguments import AlgorithmArguments
from .drq import RandomShiftsAug


@dataclasses.dataclass
class DrQv2(AlgorithmArguments):

    """
    <a href="https://github.com/learnables/cherry/blob/master/cherry/algorithms/drqv2.py" class="source-link">[Source]</a>

    ## Description

    Utilities to implement DrQ-v2 from [1].

    DrQ-v2 builds on DrQ but replaces the underlying SAC with TD3.
    It is noticeably faster in terms of wall-clock time and sample complexity.

    ## References

    1. Yarats et al., "Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning", ICLR 2022.

    ## Arguments

    * `batch_size` (int, *optional*, default=512) - Number of samples to get from the replay.
    * `discount` (float, *optional*, default=0.99) - Discount factor.
    * `policy_delay` (int, *optional*, default=1) - Delay between policy updates.
    * `target_delay` (int, *optional*, default=1) - Delay between action value updates.
    * `target_polyak_weight` (float, *optional*, default=0.995) - Weight factor `alpha` for Polyak averaging;
        see [cherry.models.polyak_average](/api/cherry.models/#cherry.models.utils.polyak_average).
    * `nsteps` (int, *optional*, default=1) - Number of bootstrapping steps to compute the target values.
    * `std_decay` (float, *optional*, default=0.0) - Exponential decay rate of the policy's standard deviation. A reasonable value for DMC is 0.99997.
    * `min_std` (float, *optional*, default=0.1) - Minimum standard deviation for the policy.
    """

    batch_size: int = 512
    discount: float = 0.99
    policy_delay: int = 1
    target_delay: int = 1
    target_polyak_weight: float = 0.995
    nsteps: int = 1
    std_decay: float = 0.0  # decent value: 0.99997
    min_std: float = 0.1

    def update(
        self,
        replay,
        policy,
        action_value,
        target_action_value,
        features,
        policy_optimizer,
        action_value_optimizer,
        features_optimizer,
        update_policy=True,
        update_target=True,
        update_value=True,
        augmentation_transform=None,
        device=None,
        **kwargs,
    ):
        """
        ## Description

        Implements a single DrQ-v2 update.

        ## Arguments

        * `replay` (cherry.ExperienceReplay) - Offline replay to sample transitions from.
        * `policy` (cherry.nn.Policy) - Policy to optimize.
        * `action_value` (cherry.nn.ActionValue) - Twin action value to optimize; see cherry.nn.Twin.
        * `target_action_value` (cherry.nn.ActionValue) - Target action value.
        * `features` (torch.nn.Module) - Feature extractor for the policy and action value.
        * `policy_optimizer` (torch.optim.Optimizer) - Optimizer for the `policy`.
        * `features_optimizer` (torch.optim.Optimizer) - Optimizer for the `features`.
        * `update_policy` (bool, *optional*, default=True) - Whether to update the policy.
        * `update_target` (bool, *optional*, default=True) - Whether to update the action value target network.
        * `update_value` (bool, *optional*, default=True) - Whether to update the action value.
        * `augmentation_transform` (torch.nn.Module, *optional*, default=None) - Data augmentation transform to augment image observations.
            Defaults to `RandomShiftsAug(4)` (as in the paper).
        * `device` (torch.device) - The device used to compute the update.
        """

        # Log debugging values
        stats = {}

        # unwrap hyper-parameters
        config = self.unpack_config(self, kwargs)

        if augmentation_transform is None:
            augmentation_transform = RandomShiftsAug(4)

        # fetch batch
        batch = replay.sample(config.batch_size, nsteps=config.nsteps, discount=config.discount)
        states = batch.state().to(device, non_blocking=True).float()
        next_states = batch.next_state().to(device, non_blocking=True).float()
        actions = batch.action().to(device, non_blocking=True)
        rewards = batch.reward().to(device, non_blocking=True)
        dones = batch.done().to(device, non_blocking=True)

        # Process states
        states = augmentation_transform(states)
        next_states = augmentation_transform(next_states)
        if features is not None:
            states = features(states)
            next_states = features(next_states)

        # Update Policy
        if update_policy:
            new_actions = policy(states.detach()).rsample()
            policy_loss = - action_value(states.detach(), new_actions).mean()

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()
            stats['drqv2/policy_loss'] = policy_loss.item()

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
            features_optimizer.zero_grad()
            value_loss.backward()
            action_value_optimizer.step()
            features_optimizer.step()

            stats['drqv2/qf_loss1'] = critic_qf1_loss.item()
            stats['drqv2/qf_loss2'] = critic_qf2_loss.item()
            stats['drqv2/batch_rewards'] = rewards.mean().item()

        # Move target approximator parameters towards critic
        if update_target:
            cherry.models.polyak_average(
                source=target_action_value,
                target=action_value,
                alpha=config.target_polyak_weight,
            )

        return stats
