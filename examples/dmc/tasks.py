# -*- coding=utf-8 -*-

"""
TODO:
    - Rewrite with metaschool.
"""

import gym
import torch
import cherry
import dataclasses
import metaschool as ms
import dmc2gym

from utils import FrameStack


ACTION_REPEATS = {
    'cartpole': 8,
    'reacher': 4,
    'cheetah': 4,
    'finger': 2,
    'ball_in_cup': 4,
    'walker': 2,
    'pendulum': 2,
    'quadruped': 2,
    'humanoid': 2,
    'hopper': 2,
}

DMC_TASKLIST = [
    ('ball_in_cup', 'catch'),
    ('cartpole', 'swingup'),
    ('finger', 'spin'),
    ('cheetah', 'run'),
    ('reacher', 'easy'),
    ('walker', 'walk'),
    ('hopper', 'stand'),
]


@dataclasses.dataclass
class DMCTasks(ms.EnvFactory, cherry.algorithms.arguments.AlgorithmArguments):

    """
    Utility class to instantiate a DMC environment.

    Largely inspired from Dr.Q's code.
    """

    domain_name: str = 'cartpole'
    task_name: str = 'swingup'
    img_size: int = 84
    action_repeat: int = -1
    frame_stack: int = 3
    time_aware: bool = False
    goal_observable: bool = True
    scale_rewards: float = 1.0
    normalize_rewards: bool = False
    max_horizon: int = -1
    camera_views: str = '13'
    grayscale: bool = False
    seed: int = 42
    vision_states: bool = True
    device: str = 'cpu'
    num_envs: int = 1

    def make(self, config=None):

        if config is None:
            config = self.sample()

        # Instantiate environment
        camera_id = 2 if config.domain_name == 'quadruped' else 0
        if config.action_repeat == -1:
            if config.domain_name in ACTION_REPEATS:
                config.action_repeat = ACTION_REPEATS[config.domain_name]
            else:
                config.action_repeat = 0

        def make_env(rank):
            def _thunk():
                env = dmc2gym.make(
                    domain_name=config.domain_name,
                    task_name=config.task_name,
                    seed=config.seed + rank,
                    visualize_reward=False,
                    from_pixels=config.vision_states,
                    height=config.img_size,
                    width=config.img_size,
                    frame_skip=config.action_repeat,
                    camera_id=camera_id,
                )
                if config.time_aware:
                    assert not config.vision_states, \
                        'time_aware not compatible w/ vision'
                    env = gym.wrappers.TimeAwareObservation(env)
                if config.frame_stack > 0:
                    env = FrameStack(env, k=config.frame_stack)
                if rank == 0:
                    env = cherry.wrappers.Logger(env, interval=1000)
                env = gym.wrappers.TransformReward(
                    env=env,
                    f=lambda r: r / config.scale_rewards,
                )
                if config.normalize_rewards:
                    env = cherry.wrappers.RewardNormalizer(env)
                env = cherry.wrappers.ActionSpaceScaler(env)
                assert env.action_space.low.min() >= -1
                assert env.action_space.high.max() <= 1
                return env
            return _thunk

        if config.num_envs == 1:
            env = make_env(0)()
        else:
            envs = [make_env(i) for i in range(config.num_envs)]
            env = gym.vector.AsyncVectorEnv(envs)

        env = cherry.wrappers.Torch(env, device=config.device)
        if config.vision_states:
            env = cherry.wrappers.StateLambda(env, lambda s: s.to(torch.uint8))
        env = cherry.wrappers.Runner(env)
        env.seed(config.seed)
        return env

    def sample(self, **kwargs):
        return ms.TaskConfig(
            domain_name=kwargs.get('domain_name', self.domain_name),
            task_name=kwargs.get('task_name', self.task_name),
            img_size=kwargs.get('img_size', self.img_size),
            action_repeat=kwargs.get('action_repeat', self.action_repeat),
            frame_stack=kwargs.get('frame_stack', self.frame_stack),
            time_aware=kwargs.get('time_aware', self.time_aware),
            goal_observable=kwargs.get('goal_observable', self.goal_observable),
            scale_rewards=kwargs.get('scale_rewards', self.scale_rewards),
            normalize_rewards=kwargs.get('normalize_rewards', self.normalize_rewards),
            max_horizon=kwargs.get('max_horizon', self.max_horizon),
            camera_views=kwargs.get('camera_views', self.camera_views),
            grayscale=kwargs.get('grayscale', self.grayscale),
            seed=kwargs.get('seed', self.seed),
            vision_states=kwargs.get('vision_states', self.vision_states),
            device=kwargs.get('device', self.device),
            num_envs=kwargs.get('num_envs', self.num_envs),
        )


    # def make_env(
    #     self,
    #     domain_name=None,
    #     task_name=None,
    #     seed=None,
    #     img_size=None,
    #     action_repeat=None,
    #     frame_stack=None,
    #     time_aware=None,
    #     goal_observable=None,
    #     scale_rewards=None,
    #     normalize_rewards=None,
    #     max_horizon=None,
    #     camera_views=None,
    #     flip_observations=None,
    #     grayscale=None,
    #     vision_states=None,
    #     num_envs=None,
    #     logger=None,
    #     device=None,
    # ):
    #     # Parse args
    #     domain_name = self.domain_name if domain_name is None else domain_name
    #     task_name = self.task_name if task_name is None else task_name
    #     seed = self.seed if seed is None else seed
    #     img_size = self.img_size if img_size is None else img_size
    #     action_repeat = self.action_repeat if action_repeat is None else action_repeat
    #     frame_stack = self.frame_stack if frame_stack is None else frame_stack
    #     time_aware = self.time_aware if time_aware is None else time_aware
    #     goal_observable = self.goal_observable if goal_observable is None else goal_observable
    #     scale_rewards = self.scale_rewards if scale_rewards is None else scale_rewards
    #     normalize_rewards = self.normalize_rewards if normalize_rewards is None else normalize_rewards
    #     max_horizon = self.max_horizon if max_horizon is None else max_horizon
    #     camera_views = self.camera_views if camera_views is None else camera_views
    #     flip_observations = self.flip_observations if flip_observations is None else flip_observations
    #     grayscale = self.grayscale if grayscale is None else grayscale
    #     vision_states = self.vision_states if vision_states is None else vision_states
    #     num_envs = self.num_envs if num_envs is None else num_envs
    #     logger = self.logger if logger is None else logger
    #     device = self.device if device is None else device
    #
    #     # Instantiate environment
    #     camera_id = 2 if domain_name == 'quadruped' else 0
    #     if action_repeat == -1:
    #         if domain_name in ACTION_REPEATS:
    #             action_repeat = ACTION_REPEATS[domain_name]
    #         else:
    #             action_repeat = 0
    #
    #     if domain_name in ACTION_REPEATS.keys():
    #         def make_base_env(rank):  # DMC Benchmark
    #             env = dmc2gym.make(
    #                 domain_name=domain_name,
    #                 task_name=task_name,
    #                 seed=seed+rank,
    #                 visualize_reward=False,
    #                 from_pixels=vision_states,
    #                 height=img_size,
    #                 width=img_size,
    #                 frame_skip=action_repeat,
    #                 camera_id=camera_id,
    #             )
    #             return env
    #     else:
    #         raise 'Unknown domain name'
    #
    #     def make_env(rank):
    #         def _thunk():
    #             env = make_base_env(rank)
    #             if time_aware:
    #                 assert not vision_states, \
    #                     'time_aware not compatible w/ vision'
    #                 env = gym.wrappers.TimeAwareObservation(env)
    #             if frame_stack > 0:
    #                 env = FrameStack(env, k=frame_stack)
    #             if rank == 0:
    #                 if logger == 'visdom':
    #                     env = cherry.wrappers.VisdomLogger(env, interval=1000)
    #                 elif logger == 'text':
    #                     env = cherry.wrappers.Logger(env, interval=1000)
    #             env = gym.wrappers.TransformReward(env, f=lambda r: r / scale_rewards)
    #             if normalize_rewards:
    #                 env = cherry.wrappers.RewardNormalizer(env)
    #             env = cherry.wrappers.ActionSpaceScaler(env)
    #             assert env.action_space.low.min() >= -1
    #             assert env.action_space.high.max() <= 1
    #             return env
    #         return _thunk
    #
    #     if num_envs == 1:
    #         env = make_env(0)()
    #     else:
    #         envs = [make_env(i) for i in range(num_envs)]
    #         env = gym.vector.AsyncVectorEnv(envs)
    #
    #     env = cherry.wrappers.Torch(env, device=device)
    #     if vision_states:
    #         env = cherry.wrappers.StateLambda(env, lambda s: s.to(torch.uint8))
    #     if flip_observations == 'hflip':
    #         env = cherry.wrappers.StateLambda(env, lambda s: torch.flip(s, dims=(-1, )))
    #     elif flip_observations == 'vflip':
    #         env = cherry.wrappers.StateLambda(env, lambda s: torch.flip(s, dims=(-2, )))
    #     elif flip_observations != '':
    #         raise ValueError
    #     env = cherry.wrappers.Runner(env)
    #     env.seed(seed)
    #     return env
