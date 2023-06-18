# -*- coding=utf-8 -*-

import dataclasses
import argparse
import gym
import numpy as np
import wandb

from collections import deque


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def evaluate_policy(
    env,
    policy,
    num_episodes,
    step=0,
    render=False,
    log_wandb=False,
):
    test_rewards = 0.0
    video = []
    for episode in range(num_episodes):
        ep_reward = 0.0
        state = env.reset()
        while True:
            if episode == 0 and render:
                frame = env.render(mode='rgb_array')
                frame = frame.transpose(2, 0, 1)
                video.append(np.expand_dims(frame, 0))
            action = policy(state)
            state, reward, done, info = env.step(action)
            ep_reward += reward
            if done:
                break
        test_rewards += ep_reward
    test_rewards /= float(num_episodes)

    if log_wandb:
        stats = {
            'test_rewards': test_rewards,
        }
        if render:
            video = np.concatenate(video, 0)
            stats['render'] = wandb.Video(video, fps=8, format='gif')
        stats['eval_step'] = step
        wandb.log(stats)
    return test_rewards


def flatten_config(args, prefix=None):
    flat_args = dict()
    if isinstance(args, argparse.Namespace):
        args = vars(args)
        return flatten_config(args)
    elif not dataclasses.is_dataclass(args) and not isinstance(args, dict):
        flat_args[prefix] = args
        return flat_args
    elif dataclasses.is_dataclass(args):
        keys = dataclasses.fields(args)
        def getvalue(x): return getattr(args, x.name)
    elif isinstance(args, dict):
        keys = args.keys()
        def getvalue(x): return args[x]
    else:
        raise 'Unknown args'
    for key in keys:
        value = getvalue(key)
        if prefix is None:
            if isinstance(key, str):
                prefix_child = key
            elif isinstance(key, dataclasses.Field):
                prefix_child = key.name
            else:
                raise 'Unknown key'
        else:
            prefix_child = prefix + '.' + key.name
        flat_child = flatten_config(value, prefix=prefix_child)
        flat_args.update(flat_child)
    return flat_args


def tie_weights(source, target):
    # TODO: Use a memo in case modules / parameters / buffers appear twice.
    # tie parameters
    for name in source._parameters.keys():
        target._parameters[name] = source._parameters[name]

    # tie buffers
    for name in source._buffers.keys():
        target._buffers[name] = source._buffers[name]

    # recurse:
    for name in source._modules.keys():
        tie_weights(source._modules[name], target._modules[name])

    # verify it worked
    for src, tgt in zip(source.parameters(), target.parameters()):
        assert src is tgt, 'Tying parameters failed.'


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype)
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


class JointOptimizer(object):

    def __init__(self, *optimizers):
        self.optimizers = optimizers

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self, *args, **kwargs):
        for opt in self.optimizers:
            opt.step(*args, **kwargs)
