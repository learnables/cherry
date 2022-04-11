# -*- coding=utf-8 -*-

# keep
from .base_wrapper import Wrapper
from .runner_wrapper import Runner
from .torch_wrapper import Torch
from .timestep_wrapper import AddTimestep
from .visdom_logger_wrapper import VisdomLogger
from .action_space_scaler_wrapper import ActionSpaceScaler

# eventually discard
from .logger_wrapper import Logger
from .openai_atari_wrapper import OpenAIAtari
from .reward_clipper_wrapper import RewardClipper
from .recorder_wrapper import Recorder
from .normalizer_wrapper import Normalizer
from .state_normalizer_wrapper import StateNormalizer
from .reward_normalizer_wrapper import RewardNormalizer
from .state_lambda_wrapper import StateLambda
from .action_lambda_wrapper import ActionLambda

# monkey-patch old functionalities
import cherry

setattr(cherry.envs, 'Wrapper', Wrapper)
setattr(cherry.envs, 'Runner', Runner)
setattr(cherry.envs, 'Torch', Torch)
setattr(cherry.envs, 'AddTimestep', AddTimestep)
setattr(cherry.envs, 'VisdomLogger', VisdomLogger)
setattr(cherry.envs, 'ActionSpaceScaler', ActionSpaceScaler)
setattr(cherry.envs, 'Logger', Logger)
