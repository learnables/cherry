# -*- coding=utf-8 -*-

"""
File: train.py
Author: Seb Arnold - seba1511.net
Email: smr.arnold@gmail.com
Description:
The high-level training script for DM-Control experiments,
with the training loop more similar to spinning-up.
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

from models import DMCPolicy, DMCQValue, DMCFeatures
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
        tags = None if args.options.tags == '' else args.options.tags.split(',')
        tags = [t for t in tags if not t == '']
        wandb.init(
            project='cherry-dmc',
            name=args.tasks.domain_name+'-'+args.tasks.task_name,
            tags=tags,
            config=flatten_config(args),
        )

    # Instantiate the task(s)
    taskset = DMCTasks(
        domain_name=args.tasks.domain_name,
        task_name=args.tasks.task_name,
        seed=args.tasks.seed,
        img_size=args.tasks.img_size,
        action_repeat=args.tasks.action_repeat,
        frame_stack=args.tasks.frame_stack,
        time_aware=args.tasks.time_aware,
        goal_observable=args.tasks.goal_observable,
        scale_rewards=args.tasks.scale_rewards,
        normalize_rewards=args.tasks.normalize_rewards,
        max_horizon=args.tasks.max_horizon,
        camera_views=args.tasks.camera_views,
        grayscale=args.tasks.grayscale,
        vision_states=args.tasks.vision_states,
        num_envs=1,
    )
    task = taskset.make()
    test_task = taskset.make(taskset.sample(
        num_envs=1,
        scale_rewards=1.0,
        normalize_rewards=False,
    ))
    __import__('pdb').set_trace()
    if isinstance(test_task, cherry.envs.Runner):
        test_task = test_task.env

    # Instantiate the learning agent
    if args.tasks.vision_states:
        observation_shape = task.observation_space.shape
        obs_channels = observation_shape[0]
        features = DMCFeatures(
            input_size=obs_channels,
            output_size=args.features.output_size,
            activation=args.features.activation,
            device=device,
            weight_path=args.features.weight_path,
            backbone=args.features.backbone,
            freeze=args.features.freeze,
        )
        features.load_weights()
        features.to(device)
        if args.options.spectral_features == 'projector':
            features.projector = torch.nn.utils.spectral_norm(features.projector)
        if args.options.progressive_features == 'ancestral':
            ancest_args = flatten_config(args.ancest)
            features.convolutions = Ancestral(
                features.convolutions,
                device=device,
                **ancest_args,
            )
            features.projector = Ancestral(
                features.projector,
                device=device,
                **ancest_args,
            )
            for p in features.convolutions.ancestral_parameters():
                p.requires_grad = True
            for p in features.projector.ancestral_parameters():
                p.requires_grad = True
        elif args.options.progressive_features == 'twin':
            twin = lambda module: ml.MetaModule(
                module,
                transforms={
                    torch.nn.Conv2d: lambda conv: ml.TwinLayer(
                        module=conv,
                        alpha=args.options.progressive_alpha,
                        reinit=args.options.progressive_reinit,
                    ),
                    torch.nn.Linear: lambda linear: ml.TwinLayer(
                        module=linear,
                        alpha=args.options.progressive_alpha,
                        reinit=args.options.progressive_reinit,
                    ),
                },
                freeze_module=True,
            )
            features.convolutions = twin(features.convolutions)
            features.projector = twin(features.projector)
            features.to(device)
        elif 'varnish' in args.options.progressive_features:
            if args.options.progressive_features == 'varnish':
                features.convolutions = varnish.varnish(features.convolutions)
                features.projector = varnish.varnish(features.projector)
            elif args.options.progressive_features == 'log_varnish':
                features.convolutions = varnish.log_varnish(features.convolutions)
                features.projector = varnish.log_varnish(features.projector)
            elif args.options.progressive_features == 'bias_varnish':
                features.convolutions = varnish.bias_varnish(features.convolutions)
                features.projector = varnish.bias_varnish(features.projector)
            elif args.options.progressive_features == 'bias_log_varnish':
                features.convolutions = varnish.bias_log_varnish(features.convolutions)
                features.projector = varnish.bias_log_varnish(features.projector)
            elif args.options.progressive_features[-1] in '0123456789':
                for upto in range(int(args.options.progressive_features[-1])):
                    features.convolutions[upto] = varnish.varnish(features.convolutions[upto])
                    # don't varnish projector with Varnish-Upto
            features.to(device)
        elif 'log_scale' in args.options.progressive_features:
            if args.options.progressive_features == 'log_scale':
                features.convolutions = varnish.log_scale(features.convolutions)
                features.projector = varnish.log_scale(features.projector)
            elif args.options.progressive_features[-1] in '0123456789':
                for upto in range(int(args.options.progressive_features[-1])):
                    features.convolutions[upto] = varnish.log_scale(features.convolutions[upto])
                    # don't varnish projector with Varnish-Upto
            features.to(device)
        elif args.options.progressive_features == 'sb-adapter':
            adapter = lambda module: ml.MetaModule(
                module,
                transforms={
                    torch.nn.Conv2d: lambda conv: torch.nn.Sequential(
                        conv,
                        ScaleBias(shape=(conv.weight.shape[0], 1, 1)),
                    ),
                    torch.nn.Linear: lambda linear: torch.nn.Sequential(
                        linear,
                        EyeLinear(linear.weight.shape[0]),
                    ),
                },
                freeze_module=True,
            )
            features.convolutions = adapter(features.convolutions)
            features.projector = adapter(features.projector)
            features.to(device)
        elif args.options.progressive_features == True or \
             args.options.progressive_features == 'progressive':
            prog_args = flatten_config(args.prog)
            features.convolutions = Progressive(
                features.convolutions,
                device=device,
                **prog_args,
            )
            features.projector = Progressive(
                features.projector,
                device=device,
                **prog_args,
            )
            for p in features.convolutions.progressive_parameters():
                p.requires_grad = True
            for p in features.projector.progressive_parameters():
                p.requires_grad = True
        feature_size = args.features.output_size
    else:
        features = l2l.nn.Lambda(lambda x: x)
        feature_size = task.state_size

    if args.features.freeze_upto > 0:
        convolutions = features.convolutions
        if 'varnish' in args.options.progressive_features or \
           'log_scale' in args.options.progressive_features or \
           'bias_varnish' in args.options.progressive_features or \
           'bias_log_varnish' in args.options.progressive_features:
            convolutions = convolutions.wrapped_module
        # convolutions = [conv, relu, conv, relu, conv, relu, conv]
        convolutions = convolutions[:args.features.freeze_upto]
        for p in convolutions.parameters():
            p.detach_()
            p.requires_grad_(False)

    encoder = target_encoder = encoder_optimizer = None
    policy = DMCPolicy(
        env=task,
        input_size=feature_size,
        activation=args.policy.activation,
        projector_size=args.policy.projector_size,
        mlp_type=args.policy.mlp_type,
        mlp_hidden=args.policy.mlp_hidden,
        device=device,
    )
    policy.to(device)
    qvalue = DMCQValue(
        env=task,
        input_size=feature_size,
        activation=args.qvalue.activation,
        projector_size=args.qvalue.projector_size,
        mlp_type=args.qvalue.mlp_type,
        mlp_hidden=args.qvalue.mlp_hidden,
        device=device,
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
    need_unfreezing = args.options.unfreeze_after > 1 \
                      and args.options.unfreeze != '' \
                      and (args.features.freeze or not (args.options.progressive_features == '' and args.options.progressive_features == 'none'))

    # Instantiate the learning algorithm
    if args.options.algorithm == 'sac':
        algorithm_update = SAC(
            batch_size=args.sac.batch_size,
            target_polyak_weight=args.sac.target_polyak_weight,
        )
    elif args.options.algorithm == 'drq':
        algorithm_update = DrQ(
            batch_size=args.drq.batch_size,
            target_polyak_weight=args.drq.target_polyak_weight,
        )
    elif args.options.algorithm == 'drqf':
        algorithm_update = DrQFast(
            batch_size=args.drq.batch_size,
            target_polyak_weight=args.drq.target_polyak_weight,
            num_iters=args.drqv2f.num_iters,
            nsteps=args.drqv2f.nsteps,
            use_automatic_entropy_tuning=args.drqv2f.use_automatic_entropy_tuning,
        )
    elif args.options.algorithm == 'drqv2':
        algorithm_update = DrQv2(
            batch_size=args.drq.batch_size,
            target_polyak_weight=args.drq.target_polyak_weight,
        )
    elif args.options.algorithm == 'drqv2f':
        algorithm_update = DrQv2Fast(**args.drqv2f)
        if args.drqv2f.std_decay > 0.0:
            policy.std = 1.0
    elif args.options.algorithm == 'drqv2fssl':
        algorithm_update = DrQv2FastSSL(**args.drqv2fssl)
        if args.drqv2fssl.std_decay > 0.0:
            policy.std = 1.0
        encoder = moco_clr.EncoderProjector(
            features=features,
            size=feature_size,
            project=True,
        )
        encoder.to(device)
        target_encoder = moco_clr.EncoderProjector(
            features=copy.deepcopy(features),
            size=feature_size,
            project=False,
        )
        target_encoder.to(device)
    elif args.options.algorithm == 'drqv2fpissl':
        algorithm_update = DrQv2FastPiSSL(**args.drqv2fpissl)
        if args.drqv2fpissl.std_decay > 0.0:
            policy.std = 1.0
        if algorithm_update.formulation == 'curl-ema':
            encoder = moco_clr.EncoderProjector(
                features=copy.deepcopy(features),
                size=feature_size,
                project=True,
                projector_layers=1,
            )
        else:
            encoder = PiSSLEncoder(
                size=feature_size,
                project=True,
                projector_layers=args.drqv2fpissl.projector_layers,
            )
        encoder.to(device)
    elif args.options.algorithm == 'drqv2fspr':
        algorithm_update = DrQv2FastSPR(**args.drqv2fspr)
        if args.drqv2fspr.std_decay > 0.0:
            policy.std = 1.0
        encoder = SPREncoder(
            state_size=feature_size,
            action_size=task.action_size,
            features=features,
            projector=PiSSLEncoder(size=feature_size, project=True, projector_layers=1),
            predictor=PiSSLEncoder(size=feature_size, project=True, projector_layers=1),
        )
        encoder.to(device)
    elif args.options.algorithm == 'sacssl':
        algorithm_update = SACSSL(**args.sacssl)
        encoder = moco_clr.EncoderProjector(
            features=features,
            size=feature_size,
            project=True,
        )
        encoder.to(device)
        target_encoder = moco_clr.EncoderProjector(
            features=copy.deepcopy(features),
            size=feature_size,
            project=False,
        )
        target_encoder.to(device)
    elif args.options.algorithm == 'drqv3f':
        algorithm_update = DrQv3Fast(**args.drqv2f)
        if args.drqv2f.std_decay > 0.0:
            policy.std = 1.0
    elif args.options.algorithm == 'drqnpgf':
        algorithm_update = DrQNPGFast(**args.drqv2f)
    elif args.options.algorithm == 'drq2npgf':
        algorithm_update = DrQv2NPGFast(**args.drqv2f)
        if args.drqv2f.std_decay > 0.0:
            policy.std = 1.0
    elif args.options.algorithm == 'drqnpg':
        algorithm_update = DrQNPG(
            batch_size=args.drq.batch_size,
            target_polyak_weight=args.drq.target_polyak_weight,
            policy_algorithm=args.drqnpg.policy_algorithm,
            num_iters=args.drqnpg.num_iters,
        )
    elif args.options.algorithm == 'drqnpg2':
        algorithm_update = DrQNPG2(
            batch_size=args.drq.batch_size,
            target_polyak_weight=args.drq.target_polyak_weight,
            policy_algorithm=args.drqnpg.policy_algorithm,
            num_iters=args.drqnpg.num_iters,
        )
    elif args.options.algorithm == 'lstdq':
        algorithm_update = LSTDQ(**args.lstdq)
        if args.lstdq.features == 'none' or args.lstdq.features == '':
            lstdq_input_size = task.action_size + feature_size
            lstdq_features = lambda x: x
        elif args.lstdq.features == 'rbf':
            lstdq_input_size = args.lstdq.num_features
            lstdq_features = rbf.RBF(
                in_features=task.action_size + feature_size,
                out_features=lstdq_input_size,
                basis_func=rbf.linear,
            )
        elif args.lstdq.features == 'rks':
            lstdq_input_size = args.lstdq.num_features
            lstdq_features = RBFSampler(
                input_size=task.action_size + feature_size,
                output_size=lstdq_input_size,
            )
        qvalue = LSTDQValue(
            input_size=lstdq_input_size,
            features=lstdq_features,
        )
        qvalue.to(device)
        qtarget = copy.deepcopy(qvalue)

    # Checkpointing
    best_rewards = - float('inf')
    def checkpoint(iteration, rewards, save_wandb=False):
        run_id = wandb.run.id if args.options.log_wandb else 0
        archive = {
            'policy': copy.deepcopy(policy).cpu().state_dict(),
            'features': copy.deepcopy(features).cpu().state_dict(),
            'qvalue': copy.deepcopy(qvalue).cpu().state_dict(),
            'qtarget': copy.deepcopy(qtarget).cpu().state_dict(),
            'iteration': iteration,
            'config': flatten_config(args),
        }
        archive_path = f'saved_models/dmc_rl/vision{args.tasks.vision_states}/{args.tasks.domain_name}-{args.tasks.task_name}/'
        archive_name = f'{run_id}_test{rewards:.2f}_lr{args.options.learning_rate}_{args.options.tags}_seed{args.options.seed}.pth'
        os.makedirs(archive_path, exist_ok=True)
        archive_full_path = os.path.join(archive_path, archive_name)
        torch.save(archive, archive_full_path)
        if save_wandb and args.options.log_wandb:
            wandb.save(archive_full_path)
            print('Weights saved in:', archive_full_path)

    # Warmup phase
    random_policy = lambda state: task.action_space.sample()
    warmup_steps = args.options.warmup
    replay = task.run(random_policy, steps=warmup_steps).to('cpu')
    for sars in replay:
        sars.done.mul_(0.0)

    # Add a feature normalizer
    if args.options.feature_normalizer == 'warmup':
        features.fit_normalizer(replay, 'warmup')
    elif args.options.feature_normalizer == 'l2':
        features.fit_normalizer(replay, 'l2')
    elif args.options.feature_normalizer == 'layernorm':
        features.fit_normalizer(replay, 'layernorm')

    # instantiate optimizers
    features_optimizer = torch.optim.Adam(
        params=set(
            list(features.parameters()) \
                + [torch.empty(1, requires_grad=True)]  # to circumvent "empty param list for states"
        ),
        lr=args.options.features_learning_rate,
        eps=args.options.features_adam_eps,
    )
    if encoder is not None:
        encoder_optimizer = torch.optim.Adam(
            encoder.parameters(),
            lr=args.options.features_learning_rate,
            eps=args.options.features_adam_eps,
        )
    policy_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=args.options.learning_rate,
        eps=args.options.adam_eps,
    )
    value_optimizer = torch.optim.Adam(
        qvalue.parameters(),
        lr=args.options.learning_rate,
        eps=args.options.adam_eps,
    )
    alpha_optimizer = torch.optim.Adam(
        [log_alpha],
        lr=args.options.learning_rate,
        eps=args.options.adam_eps,
    )

    # Warmup heads
    set_lr(features_optimizer, 0.0)
    for update in range(args.options.warmup_updates):
        stats = algorithm_update(
            replay=replay,
            policy=policy,
            features=features,
            qvalue=qvalue,
            encoder=encoder,
            target_encoder=target_encoder,
            target_value=qtarget,
            target_features=target_features,
            log_alpha=log_alpha,
            target_entropy=target_entropy,
            policy_optimizer=policy_optimizer,
            features_optimizer=features_optimizer,
            value_optimizer=value_optimizer,
            alpha_optimizer=alpha_optimizer,
            update_policy=False,
            update_target=True,
            device=device,
        )
    set_lr(features_optimizer, args.options.features_learning_rate)

    # Learning phase
    behaviour_policy = lambda x: policy.act(features(x))
    test_policy = lambda x: policy.act(
        features(x),
        deterministic=args.options.eval_deter,
    )
    total_steps = args.options.num_updates
    for step in tqdm.trange(warmup_steps, total_steps):

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
                update_policy = (true_step + update) % args.sac.policy_delay == 0
                update_target = (true_step + update) % args.sac.target_delay == 0
                stats = algorithm_update(
                    replay=replay,
                    policy=policy,
                    features=features,
                    qvalue=qvalue,
                    target_value=qtarget,
                    target_features=target_features,
                    encoder=encoder,
                    target_encoder=target_encoder,
                    log_alpha=log_alpha,
                    target_entropy=target_entropy,
                    policy_optimizer=policy_optimizer,
                    features_optimizer=features_optimizer,
                    value_optimizer=value_optimizer,
                    alpha_optimizer=alpha_optimizer,
                    encoder_optimizer=encoder_optimizer,
                    update_policy=update_policy,
                    update_target=update_target,
                    device=device,
                )

                if args.options.log_wandb:
                    stats['rl_step'] = true_step + update
                    wandb.log(stats)

        # test
        true_step += args.tasks.num_envs
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
                if true_step % args.options.checkpoint_freq == 0:
                    checkpoint(iteration=step, rewards=test_rewards)
                    best_rewards = test_rewards

        # unfreeze if necessary
        if args.options.unfreeze_after <= step and need_unfreezing:
            need_unfreezing = False
            print('Unfrozen at', step)
            # unfreeze features
            if args.options.unfreeze == 'unfreeze':
                features.unfreeze_weights()
            elif args.options.unfreeze == 'varnish':
                features.unfreeze_weights()
                features.convolutions = varnish.varnish(features.convolutions)
                features.projector = varnish.varnish(features.projector)
                features.to(device)
            target_features = copy.deepcopy(features)

            # reinit features optimizer
            features_optimizer = torch.optim.Adam(
                list(features.parameters()) + [torch.empty(1, requires_grad=True)],  # to circumvent "empty param list for states"
                lr=args.options.features_learning_rate,
                eps=args.options.features_adam_eps,
            )

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
    if args.options.save_final_replay:
        replay.to('cpu')
        run_id = wandb.run.id if args.options.log_wandb else 0
        replay_path = f'saved_replays/dmc_rl/vision{args.tasks.vision_states}/{args.tasks.domain_name}-{args.tasks.task_name}/'
        replay_name = f'{run_id}_test{test_rewards:.2f}_lr{args.options.learning_rate}_{args.options.tags}_seed{args.options.seed}.pth'
        os.makedirs(replay_path, exist_ok=True)
        replay_full_path = os.path.join(replay_path, replay_name)
        replay.save(replay_full_path)


if __name__ == "__main__":

    os.environ['OMP_NUM_THREADS'] = '6'
    os.environ['MKL_NUM_THREADS'] = '6'
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    @dataclasses.dataclass
    class TrainOptions:

        num_updates: int = 1000000  # Number of algorithm updates.
        warmup: int = 1000  # Length of warmup phase.
        warmup_updates: int = 0
        replay_size: int = 100000
        learning_rate: float = 1e-3
        features_learning_rate: float = 3e-4
        log_alpha: float = 0.0
        render: bool = False
        update_freq: int = 50
        eval_freq: int = 1000
        checkpoint_freq: int = 1000000
        render_freq: int = 1000000
        eval_deter: bool = False
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
    parser.add_arguments(DMCQValue.args, dest='qvalue')
    parser.add_arguments(DMCTasks, dest='tasks')
    parser.add_arguments(cherry.algorithms.SAC, dest='sac')
    parser.add_arguments(cherry.algorithms.DrQ, dest='drq')
    parser.add_arguments(cherry.algorithms.DrQv2, dest='drqv2f')
    args = parser.parse_args()
    main(args)
