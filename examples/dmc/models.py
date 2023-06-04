# -*- coding=utf-8 -*-

import os
import dataclasses
import torch
import cherry
import learn2learn as l2l


def dmc_initialization(module):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.orthogonal_(module.weight.data)
        if hasattr(module.bias, 'data'):
            module.bias.data.fill_(0.0)
    elif isinstance(module, torch.nn.Conv2d) \
            or isinstance(module, torch.nn.ConvTranspose2d):
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.orthogonal_(module.weight.data, gain)
        if hasattr(module.bias, 'data'):
            module.bias.data.fill_(0.0)


@dataclasses.dataclass
class DMCFeaturesArguments(cherry.algorithms.AlgorithmArguments):

    input_size: int = 9
    output_size: int = 50
    num_layers: int = 4
    num_filters: int = 32
    conv_output_size: int = 35
    activation: str = 'relu'
    weight_path: str = ''
    freeze: bool = False


class DMCFeatures(torch.nn.Module):

    """Proto-RL feature extractor."""

    args = DMCFeaturesArguments

    def __init__(
        self,
        input_size=DMCFeaturesArguments.input_size,
        output_size=DMCFeaturesArguments.output_size,
        num_layers=DMCFeaturesArguments.num_layers,
        num_filters=DMCFeaturesArguments.num_filters,
        conv_output_size=DMCFeaturesArguments.conv_output_size,
        activation=DMCFeaturesArguments.activation,
        weight_path=DMCFeaturesArguments.weight_path,
        freeze=DMCFeaturesArguments.freeze,
        device=None,
    ):
        super(DMCFeatures, self).__init__()
        self.device = device
        self.weight_path = weight_path
        self.freeze = freeze
        self.output_size = output_size
        self.use_normalizer = False

        self.activation = activation
        if self.activation == 'relu':
            activation = torch.nn.ReLU
        elif self.activation == 'gelu':
            activation = torch.nn.GeLU
        else:
            raise 'Unsupported activation'

        # convolutions
        convolutions = [
            torch.nn.Conv2d(input_size, num_filters, 3, stride=2),
        ]
        for _ in range(num_layers - 1):
            convolutions.append(activation())
            conv = torch.nn.Conv2d(
                num_filters,
                num_filters,
                kernel_size=3,
                stride=1,
            )
            convolutions.append(conv)
        self.convolutions = torch.nn.Sequential(*convolutions)
        self.convolutions.apply(dmc_initialization)

        # projector
        proj_input_size = num_filters * conv_output_size**2
        self.projector = torch.nn.Linear(
            proj_input_size,
            output_size,
        )
        self.projector.apply(dmc_initialization)

        # (optional) load / freeze weights
        self.load_weights()
        if self.freeze:
            self.freeze_weights()

    def load_weights(self, weight_path=None):
        if weight_path is None:
            weight_path = self.weight_path
        if not weight_path == '':
            weight_path = os.path.expanduser(weight_path)
            archive = torch.load(weight_path)
            try:
                self.load_state_dict(archive)
            except Exception:
                self.load_state_dict(archive['features'])
            self.to(self.device)

    def freeze_weights(self):
        for p in self.parameters():
            p.detach_()
            p.requires_grad = False

    def unfreeze_weights(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x):
        if self.device is not None:
            x = x.to(self.device, non_blocking=True)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x / 255.0  # async float conversion on GPU
        x = self.convolutions(x)
        x = x.view(x.size(0), -1)
        x = self.projector(x)
        return x


class DMCMLP(torch.nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=1024,
        num_hidden=2,
        activation='relu',
        device=None,
        **kwargs,
    ):
        super(DMCMLP, self).__init__()
        self.device = device
        if activation == 'relu':
            act = torch.nn.ReLU
        elif activation == 'gelu':
            act = torch.nn.GeLU
        else:
            raise 'Unsupported activation'

        layers = []
        if num_hidden == 0:
            layers.append(torch.nn.Linear(input_size, output_size))
        else:
            layers.append(torch.nn.Linear(input_size, hidden_size))
            layers.append(act())
            for _ in range(num_hidden - 1):
                layers.append(torch.nn.Linear(hidden_size, hidden_size))
                layers.append(act())
            layers.append(torch.nn.Linear(hidden_size, output_size))
        self.layers = torch.nn.Sequential(*layers)
        self.apply(dmc_initialization)

    def forward(self, x):
        if self.device is not None:
            x = x.to(self.device)
        return self.layers(x)


@dataclasses.dataclass
class DMCPolicyArguments(cherry.algorithms.AlgorithmArguments):

    input_size: int = 50
    activation: str = 'relu'
    projector_size: int = 0  # 0 means no projector
    mlp_hidden: int = 2
    weight_path: str = ''


class DMCPolicy(cherry.nn.Policy):

    """A policy for DMC tasks."""
    args = DMCPolicyArguments

    def __init__(
        self,
        env,
        input_size=DMCPolicyArguments.input_size,
        activation=DMCPolicyArguments.activation,
        projector_size=DMCPolicyArguments.projector_size,
        mlp_hidden=DMCPolicyArguments.mlp_hidden,
        weight_path=DMCPolicyArguments.weight_path,
        device=None,
    ):
        super(DMCPolicy, self).__init__()

        self.device = device
        self.input_size = input_size
        self.activation = activation
        self.projector_size = projector_size
        self.mlp_hidden = mlp_hidden
        self.weight_path = weight_path
        self.std = None

        if self.projector_size == 0:
            self.projector = l2l.nn.Lambda(lambda x: x)
            policy_input_size = self.input_size
        else:
            self.projector = torch.nn.Sequential(
                torch.nn.Linear(self.input_size, self.projector_size),
                torch.nn.LayerNorm(self.projector_size),
                #  torch.nn.Tanh(),  # DrQ uses a Tanh after LN
            )
            self.projector.apply(dmc_initialization)
            policy_input_size = self.projector_size

        self.actor = DMCMLP(
            input_size=policy_input_size,
            output_size=2 * env.action_size,
            activation=self.activation,
            num_hidden=self.mlp_hidden,
            device=self.device,
        )
        self.distribution = cherry.distributions.TanhNormal

    def load_weights(self, weight_path=None):
        if weight_path is None:
            weight_path = self.weight_path
        if not weight_path == '':
            weight_path = os.path.expanduser(weight_path)
            archive = torch.load(weight_path)
            try:
                self.load_state_dict(archive)
            except Exception:
                self.load_state_dict(archive['policy'])
            self.to(self.device)

    def forward(self, state):
        if self.device is not None:
            state = state.to(self.device)
        state = self.projector(state)
        mean, log_std = self.actor(state).chunk(2, dim=-1)
        if self.std is None:
            # clip log_std to [-10., 2]
            log_std = log_std.clamp(-10.0, 2.0)
            density = self.distribution(mean, log_std.exp())
        else:
            density = self.distribution(
                normal_mean=mean,
                normal_std=self.std * torch.ones_like(log_std)
            )
        return density


class DMCActionValue(cherry.nn.ActionValue):

    """A Q value function for DMC tasks."""

    args = DMCPolicyArguments

    def __init__(
        self,
        env,
        input_size=DMCPolicyArguments.input_size,
        activation=DMCPolicyArguments.activation,
        projector_size=DMCPolicyArguments.projector_size,
        mlp_hidden=DMCPolicyArguments.mlp_hidden,
        weight_path=DMCPolicyArguments.weight_path,
        device=None,
    ):
        super(DMCActionValue, self).__init__()

        self.device = device
        self.input_size = input_size
        self.activation = activation
        self.projector_size = projector_size
        self.mlp_hidden = mlp_hidden
        self.weight_path = weight_path

        if self.projector_size == 0:
            self.projector = l2l.nn.Lambda(lambda x: x)
            state_input_size = self.input_size
        else:
            self.projector = torch.nn.Sequential(
                torch.nn.Linear(self.input_size, self.projector_size),
                torch.nn.LayerNorm(self.projector_size),
                #  torch.nn.Tanh(),  # DrQ uses a Tanh after LN
            )
            state_input_size = self.projector_size
            self.projector.apply(dmc_initialization)

        action_size = env.action_size
        self.q1 = DMCMLP(
            input_size=state_input_size + action_size,
            output_size=1,
            num_hidden=self.mlp_hidden,
            activation=self.activation,
        )
        self.q2 = DMCMLP(
            input_size=state_input_size + action_size,
            output_size=1,
            num_hidden=self.mlp_hidden,
            activation=self.activation,
        )

    def load_weights(self, weight_path=None):
        if weight_path is None:
            weight_path = self.weight_path
        weight_path = os.path.expanduser(weight_path)
        if not weight_path == '':
            archive = torch.load(weight_path)
            try:
                self.load_state_dict(archive)
            except Exception:
                self.load_state_dict(archive['qvalue'])
            self.to(self.device)

    def forward(self, state, action):
        if self.device is not None:
            state = state.to(self.device)
            action = action.to(self.device)
        qf1, qf2 = self.twin(state, action)
        return torch.min(qf1, qf2)

    def twin(self, state, action):
        if self.device is not None:
            state = state.to(self.device)
            action = action.to(self.device)
        state = self.projector(state)
        state_value = torch.cat([state, action], dim=-1)
        q1 = self.q1(state_value)
        q2 = self.q2(state_value)
        return q1, q2
