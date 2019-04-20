#!/usr/bin/env python3

import sys
import os
import inspect

import random
import numpy as np
import torch as th

import wandb


if __name__ == '__main__':
    main = None
    script, env, seed = sys.argv[1:]
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

    # Train
    exec(main_code)

    # Compute and log all rewards
    if hasattr(env, 'all_rewards'):
        R = 0
        for reward, done in zip(env.all_rewards, env.all_dones):
            wandb.log({
                'all_rewards': reward,
                'all_dones': done,
            })
            R += reward
            if bool(done):
                wandb.log({'episode_rewards': R})
                R = 0

    # Compute some test rewards
    # Save model weights
    # Save some visualizations
