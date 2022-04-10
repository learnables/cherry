# -*- coding=utf-8 -*-

import torch
import cherry
import dataclasses
import dotmap

from .arguments import AlgorithmArguments
from .sac import SAC
from .drq import RandomShiftsAug


@dataclasses.dataclass
class DrQv2(AlgorithmArguments):

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
