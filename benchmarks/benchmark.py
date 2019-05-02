#!/usr/bin/env python3

import sys
import os
import inspect

import random
import numpy as np
import torch as th
import cherry as ch

import wandb

from shutil import copyfile


def benchmark_log(original_log):
    def new_log(self, key, value):
        wandb.log({key: value}, step=self.num_steps)
        original_log(self, key, value)
    return new_log


def benchmark_stats(original_stats):
    mean = lambda x: sum(x) / len(x)

    def new_stats(self, *args, **kwargs):
        result = original_stats(self, *args, **kwargs)
        wandb.log({'10_ep_mean_reward': mean(result['episode_rewards']),
                   '10_ep_mean_length': mean(result['episode_lengths'])},
                  step=self.num_steps)
        return result
    return new_stats


if __name__ == '__main__':
    # To avoid undefined warnings
    main = get_action = None
    envs = agent = policy = model = actor = critic = None

    # Parse arguments
    script = os.environ['BENCH_SCRIPT']
    env = os.environ['BENCH_ENV']
    seed = os.environ['BENCH_SEED']
    script_dir = os.path.dirname(script)
    script_file = os.path.basename(script)
    seed = int(seed)
    print('Benchmarks: Running ' + script)

    # Init result logger
    config = {
        'seed': seed,
        'env': env,
        'script': script,
        'cherry.__version__': ch.__version__,
    }
    wandb.init(
        project='cherry-benchmarks',
        name=script_file[:-3] + '/' + env,
        group=script,
        config=config,
    )

    # Save benchmarking script
    path = os.path.join(wandb.run.dir, script_file)
    copyfile(script, path)


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
        for i, (reward, done) in enumerate(zip(env.all_rewards,
                                               env.all_dones)):
            wandb.log({
                'all_rewards': reward,
                'all_dones': done,
            }, step=i)

            R += reward
            if bool(done):
                wandb.log({
                    'episode_rewards': R,
                }, step=i)
                returns.append(R)
                R = 0

        smoothed_returns = ch.plot.smooth(returns)
        for i, r in enumerate(smoothed_returns):
            wandb.log({
                'smoothed_returns': r,
            }, step=i)

    # Save model weights
    print('Benchmarks: Saving weights.')
    for name, __model in [('model.pth', model),
                          ('actor.pth', actor),
                          ('critic.pth', critic),
                          ('policy.pth', policy),
                          ('agent.pth', agent)]:
        if __model is not None:
            path = os.path.join(wandb.run.dir, name)
            if hasattr(__model, 'state_dict'):
                th.save(__model.state_dict(),  path)

    # Sample new episodes
    act = None
    if get_action is not None:  # PyBulet + Atari
        act = get_action
    elif agent is not None:  # Spinup + Tabular
        act = agent

    if act is not None:
        # Compute some test rewards
        if hasattr(env, 'run'):
            print('Benchmarks: Computing test rewards.')
            for step in range(25):
                replay = env.run(act, episodes=1)
                sum_rew = replay.reward().sum().item()
                wandb.log({
                    'test_time_rewards': sum_rew
                }, step=step)
        # TODO: Save some visualizations
        print('Benchmarks: Creating rollout videos.')
