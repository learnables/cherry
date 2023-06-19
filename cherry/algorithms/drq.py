# -*- coding=utf-8 -*-

import torch
import cherry
import dataclasses

from .arguments import AlgorithmArguments
from .sac import SAC


@dataclasses.dataclass
class DrQ(AlgorithmArguments):

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
    target_polyak_weight: float = 0.995

    def update(
        self,
        replay,
        policy,
        action_value,
        target_action_value,
        features,
        target_features,
        log_alpha,
        target_entropy,
        policy_optimizer,
        action_value_optimizer,
        features_optimizer,
        alpha_optimizer,
        update_policy=True,
        update_target=False,
        update_value=True,
        update_entropy=True,
        augmentation_transform=None,
        device=None,
        **kwargs,
    ):
        stats = {}

        # unwrap hyper-parameters
        config = self.unpack_config(self, kwargs)

        if augmentation_transform is None:
            augmentation_transform = RandomShiftsAug(4)

        # Sample mini-batch
        batch = replay.sample(config.batch_size)
        if device is not None:
            batch = batch.to(device)
        if batch.vectorized:
            batch = batch.flatten()
        states = batch.state().float()
        aug_states = augmentation_transform(states)
        states = augmentation_transform(states)
        if features is not None:
            states = features(states)
            aug_states = features(aug_states)

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
                alpha = log_alpha.exp()
                alpha_loss = torch.zeros(1)

            stats['drq/alpha_loss'] = alpha_loss.item()
            stats['drq/alpha_value'] = alpha.item()

        # Update Q-function
        if update_value:
            actions = batch.action()
            qf1_estimate, qf2_estimate = action_value.twin(
                states,
                actions.detach(),
            )
            aug_qf1_estimate, aug_qf2_estimate = action_value.twin(
                aug_states,
                actions.detach(),
            )

            # compute targets
            with torch.no_grad():
                next_states = batch.next_state().float()
                target_next_states = next_states
                aug_next_states = augmentation_transform(next_states)
                next_states = augmentation_transform(next_states)
                aug_target_next_states = augmentation_transform(target_next_states)
                target_next_states = augmentation_transform(target_next_states)
                if features is not None:
                    next_states = features(next_states)
                    aug_next_states = features(aug_next_states)
                if target_features is not None:
                    target_next_states = target_features(target_next_states)
                    aug_target_next_states = target_features(aug_target_next_states)

                next_density = policy(next_states)
                next_actions, next_log_probs = SAC.actions_log_probs(next_density)
                aug_next_density = policy(aug_next_states)
                aug_next_actions, aug_next_log_probs = SAC.actions_log_probs(aug_next_density)

                target_q_values = target_action_value(target_next_states, next_actions) \
                    - alpha * next_log_probs
                aug_target_q_values = target_action_value(aug_target_next_states, aug_next_actions) \
                    - alpha * aug_next_log_probs
                target_q = 0.5 * (target_q_values + aug_target_q_values)

            rewards = batch.reward().reshape(target_q_values.shape)
            dones = batch.done().reshape(target_q_values.shape)
            critic_qf1_loss = SAC.action_value_loss(
                qf1_estimate,
                target_q.detach(),
                rewards,
                dones,
                config.discount,
            ).clamp(-1, 5e4)

            critic_qf2_loss = SAC.action_value_loss(
                qf2_estimate,
                target_q.detach(),
                rewards,
                dones,
                config.discount,
            ).clamp(-1, 5e4)
            value_loss = (critic_qf1_loss + critic_qf2_loss) / 2.0

            aug_critic_qf1_loss = SAC.action_value_loss(
                aug_qf1_estimate,
                target_q.detach(),
                rewards,
                dones,
                config.discount,
            ).clamp(-1, 5e4)

            aug_critic_qf2_loss = SAC.action_value_loss(
                aug_qf2_estimate,
                target_q.detach(),
                rewards,
                dones,
                config.discount,
            ).clamp(-1, 5e4)
            value_loss += (aug_critic_qf1_loss + aug_critic_qf2_loss) / 2.0

            # Update Critic Networks
            action_value_optimizer.zero_grad()
            features_optimizer.zero_grad()
            value_loss.backward()
            action_value_optimizer.step()
            features_optimizer.step()

            stats['drq/qf_loss1'] = critic_qf1_loss.item()
            stats['drq/qf_loss2'] = critic_qf2_loss.item()
            stats['drq/batch_rewards'] = rewards.mean().item()

        # Delayed Updates
        if update_policy:
            # Policy loss
            states = batch.state()  # recompute features
            if features is not None:
                # detach features per DrQ.
                states = features(states).detach()
            density = policy(states)
            new_actions, log_probs = SAC.actions_log_probs(density)
            q_values = action_value(states, new_actions)
            policy_loss = SAC.policy_loss(log_probs, q_values, alpha.detach())

            policy_optimizer.zero_grad()
            policy_loss.clamp(-1e3, 1e3).backward()
            policy_optimizer.step()

            stats['drq/policy_entropy'] = -log_probs.mean()
            stats['drq/policy_loss'] = policy_loss.item()

        # Move target approximator parameters towards critic
        if update_target:
            cherry.models.polyak_average(
                source=target_action_value,
                target=action_value,
                alpha=config.target_polyak_weight,
            )
            if features is not None:
                cherry.models.polyak_average(
                    source=target_features,
                    target=features,
                    alpha=config.target_polyak_weight,
                )

        return stats


class RandomShiftsAug(torch.nn.Module):

    """
    Take from DrQv2 implementation:

    - Reference: Yarats et al., 2021, "Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning"
    - Code: https://github.com/facebookresearch/drqv2
    """

    def __init__(self, pad):
        super(RandomShiftsAug, self).__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = torch.nn.functional.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return torch.nn.functional.grid_sample(
            x,
            grid,
            padding_mode='zeros',
            align_corners=False)
