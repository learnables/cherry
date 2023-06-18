# -*- coding=utf-8 -*-

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
class DMCTasks(ms.EnvFactory, cherry.algorithms.AlgorithmArguments):

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
            seed=kwargs.get('seed', self.seed),
            vision_states=kwargs.get('vision_states', self.vision_states),
            device=kwargs.get('device', self.device),
            num_envs=kwargs.get('num_envs', self.num_envs),
        )
