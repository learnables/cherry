# -*- coding=utf-8 -*-

"""
File: train.py
Description:
The high-level training script for DM-Control experiments.
"""

import os
import wandb
import copy
import dataclasses
import random
import numpy as np
import torch
import cherry
import learn2learn as l2l
import simple_parsing as sp
import tqdm

from models import DMCPolicy, DMCActionValue, DMCFeatures
from tasks import DMCTasks
from utils import flatten_config, evaluate_policy, set_lr


def main(args):
    """
    Main training function for DMC experiments.
    """

    # Setup
    if args.options.debug:
        cherry.debug.debug('cherry_debug')
    random.seed(args.options.seed)
    np.random.seed(args.options.seed)
    torch.manual_seed(args.options.seed)
    torch.set_num_threads(4)
    device = 'cpu'

    if torch.cuda.is_available() and args.options.cuda:
        torch.cuda.manual_seed(args.options.seed)
        torch.backends.cudnn.benchmark = True
        device = 'cuda'

    if args.options.log_wandb:
        tags = None if args.options.tags == '' else args.options.tags.split(
            ',')
        tags = [t for t in tags if not t == '']
        wandb.init(
            project='cherry-dmc',
            name=args.tasks.domain_name+'-'+args.tasks.task_name,
            tags=tags,
            config=flatten_config(args),
        )

    # Instantiate the task(s)
    taskset = DMCTasks(**args.tasks)
    task = taskset.make()
    test_task = taskset.make(taskset.sample(
        scale_rewards=1.0,
        normalize_rewards=False,
    ))
    if isinstance(test_task, cherry.envs.Runner):
        test_task = test_task.env

    # Instantiate the learning agent
    if args.tasks.vision_states:
        args.features.input_size = task.observation_space.shape[0]
        features = DMCFeatures(
            device=device,
            **args.features,
        )
        features.to(device)
        feature_size = args.features.output_size
    else:
        features = l2l.nn.Lambda(lambda x: x)
        feature_size = task.state_size

    args.policy.input_size = feature_size
    policy = DMCPolicy(
        env=task,
        device=device,
        **args.policy,
    )
    policy.to(device)
    qvalue = DMCActionValue(
        env=task,
        device=device,
        **args.qvalue,
    )
    qvalue.to(device)
    qtarget = copy.deepcopy(qvalue)
    target_features = copy.deepcopy(features)
    log_alpha = torch.tensor(
        [args.options.log_alpha],
        requires_grad=True,
        device=device,
    )
    target_entropy = - task.action_size

    # Instantiate the learning algorithm
    if args.options.algorithm == 'sac':
        algorithm = cherry.algorithms.SAC(**args.sac)
    elif args.options.algorithm == 'drq':
        algorithm = cherry.algorithms.DrQ(**args.drq)
    elif args.options.algorithm == 'drqv2':
        algorithm = cherry.algorithms.DrQv2(**args.drqv2)
        if args.drqv2.std_decay > 0.0:
            policy.std = 1.0

    # Checkpointing
    best_rewards = - float('inf')

    def checkpoint(iteration, rewards, save_wandb=False):
        if rewards > best_rewards:
            run_id = wandb.run.id if args.options.log_wandb else 0
            archive = {
                'policy': copy.deepcopy(policy).cpu().state_dict(),
                'features': copy.deepcopy(features).cpu().state_dict(),
                'qvalue': copy.deepcopy(qvalue).cpu().state_dict(),
                'qtarget': copy.deepcopy(qtarget).cpu().state_dict(),
                'iteration': iteration,
                'config': flatten_config(args),
            }
            archive_path = f'saved_models/{args.tasks.domain_name}-{args.tasks.task_name}/{args.options.algorithm}/'
            archive_name = f'{run_id}_test{rewards:.2f}.pth'
            os.makedirs(archive_path, exist_ok=True)
            archive_full_path = os.path.join(archive_path, archive_name)
            torch.save(archive, archive_full_path)
            if save_wandb and args.options.log_wandb:
                wandb.save(archive_full_path)
            print('Weights saved in:', archive_full_path)

    # Warmup phase
    def random_policy(state): return task.action_space.sample()
    replay = task.run(random_policy, steps=args.options.warmup).to('cpu')
    for sars in replay:  # no absorbing state
        sars.done.mul_(0.0)

    # instantiate optimizers
    features_optimizer = torch.optim.Adam(
        params=set(
            list(features.parameters())
            # to circumvent "empty param list for states"
            + [torch.empty(1, requires_grad=True)]
        ),
        lr=args.options.features_learning_rate,
    )
    policy_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=args.options.learning_rate,
    )
    value_optimizer = torch.optim.Adam(
        qvalue.parameters(),
        lr=args.options.learning_rate,
    )
    alpha_optimizer = torch.optim.Adam(
        [log_alpha],
        lr=args.options.learning_rate,
    )

    # Learning phase
    def behaviour_policy(x): return policy.act(features(x))

    def test_policy(x): return policy.act(
        features(x),
        deterministic=args.options.deterministic_eval,
    )
    for step in tqdm.trange(args.options.warmup, args.options.num_updates):

        true_step = step

        # collect data
        with torch.no_grad():
            step_replay = task.run(behaviour_policy, steps=1).to('cpu')
            for sars in step_replay:
                sars.done.mul_(0.0)
            replay += step_replay
            replay = replay[-args.options.replay_size:]

        # update
        if true_step % args.options.update_freq == 0:

            for update in range(args.options.update_freq):
                stats = algorithm.update(
                    replay=replay,
                    policy=policy,
                    features=features,
                    action_value=qvalue,
                    target_action_value=qtarget,
                    target_features=target_features,
                    log_alpha=log_alpha,
                    target_entropy=target_entropy,
                    policy_optimizer=policy_optimizer,
                    features_optimizer=features_optimizer,
                    action_value_optimizer=value_optimizer,
                    alpha_optimizer=alpha_optimizer,
                    update_target=True,
                    device=device,
                )

                if args.options.log_wandb:
                    stats['rl_step'] = true_step + update
                    wandb.log(stats)
        true_step += args.tasks.num_envs

        # test
        if true_step % args.options.eval_freq == 0:
            with torch.no_grad():
                test_rewards = evaluate_policy(
                    env=test_task,
                    policy=test_policy,
                    num_episodes=10,
                    step=true_step,
                    render=true_step % args.options.render_freq == 0,
                    log_wandb=args.options.log_wandb,
                )
                best_rewards = max(test_rewards, best_rewards)
                if true_step % args.options.checkpoint_freq == 0:
                    checkpoint(iteration=step, rewards=test_rewards)

    # Save results to wandb
    with torch.no_grad():
        test_rewards = evaluate_policy(
            env=test_task,
            policy=test_policy,
            num_episodes=10,
            step=true_step,
            render=true_step % args.options.render_freq == 0,
            log_wandb=args.options.log_wandb,
        )
    checkpoint(iteration=step, rewards=test_rewards, save_wandb=True)


if __name__ == "__main__":

    os.environ['OMP_NUM_THREADS'] = '6'
    os.environ['MKL_NUM_THREADS'] = '6'
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    @dataclasses.dataclass
    class TrainOptions:

        num_updates: int = 500000  # Number of algorithm updates.
        warmup: int = 5000  # Length of warmup phase.
        replay_size: int = 100000
        learning_rate: float = 3e-3
        features_learning_rate: float = 3e-4
        log_alpha: float = 0.0
        render: bool = False
        update_freq: int = 50
        eval_freq: int = 1000
        checkpoint_freq: int = 1000000
        render_freq: int = 1000000
        deterministic_eval: bool = False
        cuda: bool = True
        log_wandb: bool = False
        debug: bool = False
        tags: str = ''
        algorithm: str = 'sac'
        seed: int = 1234

    parser = sp.ArgumentParser(add_dest_to_option_strings=True)
    parser.add_arguments(TrainOptions, dest='options')
    parser.add_arguments(DMCFeatures.args, dest='features')
    parser.add_arguments(DMCPolicy.args, dest='policy')
    parser.add_arguments(DMCActionValue.args, dest='qvalue')
    parser.add_arguments(DMCTasks, dest='tasks')
    parser.add_arguments(cherry.algorithms.SAC, dest='sac')
    parser.add_arguments(cherry.algorithms.DrQ, dest='drq')
    parser.add_arguments(cherry.algorithms.DrQv2, dest='drqv2')
    args = parser.parse_args()
    main(args)
