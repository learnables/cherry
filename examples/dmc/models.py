# -*- coding=utf-8 -*-

import os
import dataclasses
import torch
import cherry
import learn2learn as l2l
import torchvision as tv


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


class FeaturesNormalizer(torch.nn.Module):

    def __init__(self, device):
        super(FeaturesNormalizer, self).__init__()
        self.mean = torch.zeros(1)
        self.std = torch.ones(1)
        self.to(device)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return x

    def fit(self, features, replay):
        with torch.no_grad():
            bsz = 32
            device = features.device
            X = []
            for i in range(len(replay) // bsz):
                states = replay[i*bsz:(i+1)*bsz].state().to(device)
                X.append(features(states).cpu())
            X = torch.cat(X, dim=0)
            self.mean = X.mean(dim=0, keepdims=True).to(features.device)
            self.std = X.std(dim=0, keepdims=True).to(features.device)


@dataclasses.dataclass
class DMCFeaturesArguments:

    input_size: int = 9
    output_size: int = 50
    num_layers: int = 4
    num_filters: int = 32
    conv_output_size: int = 35
    activation: str = 'relu'
    backbone: str = 'dmc'
    weight_path: str = ''
    freeze: bool = False
    freeze_upto: int = 0


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
        backbone=DMCFeaturesArguments.backbone,
        device=None,
        weight_path=DMCFeaturesArguments.weight_path,
        freeze=DMCFeaturesArguments.freeze,
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

        if backbone == 'dmc':
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
            proj_input_size = num_filters * conv_output_size**2
        elif backbone == 'resnet12':
            convolutions = l2l.vision.models.ResNet12(
                output_size=1,
                channels=input_size,
            )
            self.convolutions = convolutions.features
            proj_input_size = 640  # Assumes x = (b, c, 84, 84)
        elif 'resnet18' in backbone:
            pretrained = '-pre' in backbone
            convolutions = tv.models.resnet18(pretrained=pretrained)
            convolutions = torch.nn.Sequential(  # Assumes x = (b, c, 84, 84)
                l2l.nn.Lambda(lambda x: x.reshape(-1, 3, 84, 84)),
                convolutions.conv1,
                convolutions.bn1,
                convolutions.relu,
                convolutions.maxpool,
                convolutions.layer1,
                convolutions.layer2,
                convolutions.layer3,
                convolutions.layer4,
                convolutions.avgpool,
                l2l.nn.Lambda(lambda x: torch.flatten(x, 1).reshape(-1, 512*input_size // 3))
            )
            self.convolutions = convolutions
            proj_input_size = 512 * 3
        else:
            raise 'Unknown backbone'
        self.projector = torch.nn.Linear(
            proj_input_size,
            output_size,
        )
        self.projector.apply(dmc_initialization)
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
            except:
                self.load_state_dict(archive['features'])
            self.to(self.device)

    def freeze_weights(self):
        for p in self.parameters():
            p.detach_()
            p.requires_grad = False

    def unfreeze_weights(self):
        for p in self.parameters():
            p.requires_grad = True

    def fit_normalizer(self, replay, normalizer='warmup'):
        if normalizer == 'warmup':
            self.normalizer = FeaturesNormalizer(device=self.device)
            self.normalizer.fit(self, replay)
        elif normalizer == 'layernorm':
            self.normalizer = torch.nn.LayerNorm(self.output_size)
            self.normalizer.to(self.device)
        elif normalizer == 'l2':
            self.normalizer = lambda x: 2.0 * x / x.norm(p=2, dim=1, keepdim=True)
        else:
            raise 'Unknown normalizer'
        self.use_normalizer = True

    def forward(self, x):
        if self.device is not None:
            x = x.to(self.device, non_blocking=True)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x / 255.0  # async float conversion on GPU
        x = self.convolutions(x)
        x = x.view(x.size(0), -1)
        x = self.projector(x)
        if self.use_normalizer:
            x = self.normalizer(x)
        return x


class CherryMLP(torch.nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=300,
        num_hidden=2,
        activation='relu',
        device=None,
        init_w=3e-3,
        **kwargs,
    ):
        super(CherryMLP, self).__init__()
        layer_sizes = [hidden_size, ] * num_hidden
        self.layers = torch.nn.ModuleList()
        self.device = device
        if activation == 'relu':
            self.act = torch.nn.ReLU()
        elif activation == 'gelu':
            self.act = torch.nn.GeLU()
        else:
            raise 'Unsupported activation'

        in_size = input_size
        for next_size in layer_sizes:
            fc = torch.nn.Linear(in_size, next_size)
            self.layers.append(fc)
            in_size = next_size

        self.last_fc = torch.nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, *args, **kwargs):
        h = torch.cat(args, dim=-1)
        if self.device is not None:
            h = h.to(self.device)
        for fc in self.layers:
            h = self.act(fc(h))
        output = self.last_fc(h)
        return output


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
class DMCPolicyArguments:

    input_size: int = 50
    activation: str = 'relu'
    projector_size: int = 0
    mlp_type: str = 'cherry'  # values: cherry, dmc
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
        mlp_type=DMCPolicyArguments.mlp_type,
        mlp_hidden=DMCPolicyArguments.mlp_hidden,
        device=None,
        weight_path=DMCPolicyArguments.weight_path,
    ):
        super(DMCPolicy, self).__init__()

        self.device = device
        self.input_size = input_size
        self.activation = activation
        self.projector_size = projector_size
        self.mlp_type = mlp_type
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

        mlp = DMCMLP if self.mlp_type == 'dmc' else CherryMLP
        self.actor = mlp(
            input_size=policy_input_size,
            output_size=2 * env.action_size,
            activation=self.activation,
            num_hidden=self.mlp_hidden,
            init_w=1e-3,  # only for cherry MLP class
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
            except:
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
            #  log_std = torch.tanh(log_std)
            #  log_std = -10. + 0.5 * (2. - (-10.)) * (log_std + 1)
            density = self.distribution(mean, log_std.exp())
        else:
            density = self.distribution(
                normal_mean=mean,
                normal_std=self.std * torch.ones_like(log_std)
            )
        return density


class DMCQValue(torch.nn.Module):

    """A Q value function for DMC tasks."""

    args = DMCPolicyArguments

    def __init__(
        self,
        env,
        input_size=DMCPolicyArguments.input_size,
        activation=DMCPolicyArguments.activation,
        projector_size=DMCPolicyArguments.projector_size,
        mlp_type=DMCPolicyArguments.mlp_type,
        mlp_hidden=DMCPolicyArguments.mlp_hidden,
        device=None,
        weight_path=DMCPolicyArguments.weight_path,
    ):
        super(DMCQValue, self).__init__()

        self.device = device
        self.input_size = input_size
        self.activation = activation
        self.projector_size = projector_size
        self.mlp_type = mlp_type
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
        mlp = DMCMLP if self.mlp_type == 'dmc' else CherryMLP
        self.q1 = mlp(
            input_size=state_input_size + action_size,
            output_size=1,
            num_hidden=self.mlp_hidden,
            activation=self.activation,
        )
        self.q2 = mlp(
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
            except:
                self.load_state_dict(archive['qvalue'])
            self.to(self.device)

    def forward(self, state, action):
        if self.device is not None:
            state = state.to(self.device)
            action = action.to(self.device)
        qf1, qf2 = self.twin_values(state, action)
        return torch.min(qf1, qf2)

    def twin_values(self, state, action):
        if self.device is not None:
            state = state.to(self.device)
            action = action.to(self.device)
        state = self.projector(state)
        state_value = torch.cat([state, action], dim=-1)
        q1 = self.q1(state_value)
        q2 = self.q2(state_value)
        return q1, q2
