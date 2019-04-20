#!/usr/bin/env python3

import sys
import os
import inspect

import random
import numpy as np
import torch as th
import cherry as ch

import wandb


def benchmark_log(original_log):
    def new_log(self, key, value):
        wandb.log({key: value})
        original_log(self, key, value)
    return new_log


def benchmark_stats(original_stats):
    mean = lambda x: sum(x) / len(x)

    def new_stats(self, *args, **kwargs):
        result = original_stats(self, *args, **kwargs)
        wandb.log({'last_episode_rewards': mean(result['episode_rewards'])})
        return result
    return new_stats


if __name__ == '__main__':
    # To avoid undefined warnings
    main = None
    envs = agent = policy = model = actor = critic = None

    # Parse arguments
    script, env, seed = sys.argv[-3:]
    script_dir = os.path.dirname(script)
    script_file = os.path.basename(script)
    seed = int(seed)

    # Init result logger
    config = {
        'seed': seed,
        'env': env,
        'script': script,
    }
    wandb.init(
        project='benchmarks',
        name=script_file[:-3] + '/' + env,
        group=script,
        config=config,
    )

    # Import script context
    sys.path.insert(0, script_dir)
    exec('from ' + script_file[:-3] + ' import *')
    vars_in_main = main.__code__.co_varnames
    main_code = inspect.getsourcelines(main)[0]  # 0 is code, 1 is num lines
    main_code = [line[4:] for line in main_code[1:]]  # 1 omits the fn def, 4 is for first indent
    main_code = ''.join(main_code)

    # Seed everything
    SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    # Wrap envs.Logger.log for live logging
    envs.Logger.log = benchmark_log(envs.Logger.log)
    envs.Logger._episodes_stats = benchmark_stats(envs.Logger._episodes_stats)

    # Train
    print('Benchmarks: Started training.')
    exec(main_code)

    # Update informations about environment
    if hasattr(env, 'spec'):
        wandb.config.update(env.spec)

    # Compute and log all rewards
    if hasattr(env, 'all_rewards'):
        print('Benchmarks: Computing rewards.')
        R = 0
        returns = []
        for reward, done in zip(env.all_rewards, env.all_dones):
            wandb.log({
                'all_rewards': reward,
                'all_dones': done,
            })

            R += reward
            if bool(done):
                wandb.log({
                    'episode_rewards': R,
                })
                returns.append(R)
                R = 0

        smoothed_returns = ch.plot.smooth(returns)[1]
        for r in smoothed_returns:
            wandb.log({
                'smoothed_returns': r,
            })

    # TODO: Compute some test rewards

    # Save model weights
    print('Benchmarks: Saving weights.')
    for name, __model in [('model.pth', model),
                          ('actor.pth', actor),
                          ('critic.pth', critic),
                          ('policy.pth', policy),
                          ('agent.pth', agent)]:
        if __model is not None:
            path = os.path.join(wandb.run.dir, name)
            th.save(__model.state_dict(),  path)

    # TODO: Save some visualizations
