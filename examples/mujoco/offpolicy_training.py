# -*- coding=utf-8 -*-

import copy
import random
import numpy as np
import torch
import cherry
import gym
import tqdm
import wandb


class MLPPolicy(cherry.nn.Policy):

    def __init__(self, state_size, action_size):
        super(MLPPolicy, self).__init__()
        self.mlp = cherry.nn.MLP(
            input_size=state_size,
            output_size=action_size,
            hidden_sizes=[256, 256],
            activation=torch.nn.GELU,
        )
        self.std = 0.1
        self.action_distribution = cherry.distributions.TanhNormal

    def forward(self, state):
        mean = self.mlp(state)
        std = torch.ones_like(mean) * self.std
        return self.action_distribution(mean, std)


class MLPActionValue(cherry.nn.ActionValue):

    def __init__(self, state_size, action_size):
        super(MLPActionValue, self).__init__()
        self.mlp = cherry.nn.MLP(
            input_size=state_size + action_size,
            output_size=1,
            hidden_sizes=[256, 256],
            activation=torch.nn.GELU,
        )

    def forward(self, state, action):
        return self.mlp(torch.cat([state, action], dim=1))


def main(args):

    # setup
    device = torch.device('cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    if args.cuda and torch.cuda.device_count() > 0:
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)

    wandb.init(
        entity='arnolds',
        project='dev',
        name=f'{args.env}-{args.algorithm}',
        mode='online',
    )

    # instantiate task
    def make_env():
        env = gym.make(args.env)
        env = cherry.wrappers.ActionSpaceScaler(env)
        env = cherry.wrappers.Torch(env, device=device)
        env = cherry.wrappers.Runner(env)
        env.seed(args.seed)
        return env
    env = make_env()
    eval_env = make_env()

    # instantiate policy and value
    policy = MLPPolicy(env.state_size, env.action_size)
    qvalue = cherry.nn.Twin(
        MLPActionValue(env.state_size, env.action_size),
        MLPActionValue(env.state_size, env.action_size),
    )
    target_qvalue = copy.deepcopy(qvalue)
    policy.to(device)
    qvalue.to(device)
    target_qvalue.to(device)

    # algo & optimizers
    policy_opt = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    qvalue_opt = torch.optim.Adam(qvalue.parameters(), lr=args.learning_rate)

    if args.algorithm == 'td3':
        algorithm = cherry.algorithms.TD3(
            batch_size=args.batch_size,
        )

    # train
    ep_reward = 0.0
    replay = env.run(
        lambda s: env.action_space.sample(),
        steps=1000,
    ).to(device, non_blocking=True)
    for iteration in tqdm.trange(1000, args.num_iterations, args.update_interval):

        stats = {}

        # collect data
        if iteration < args.warmup_steps:
            behavior_policy = lambda s: env.action_space.sample()
        else:
            behavior_policy = lambda s: policy.act(s)
        iter_replay = env.run(
            behavior_policy,
            steps=args.update_interval,
        )
        for sars in iter_replay:
            ep_reward += sars.reward
            if sars.done:
                stats['train/rewards'] = ep_reward
                stats['train/iteration'] = iteration
            sars.done.mul_(0.0)  # no terminal states
        replay += iter_replay.to(device, non_blocking=True)

        # update policy
        for update in range(args.update_interval):
            update_stats = algorithm.update(
                replay=replay,
                policy=policy,
                action_value=qvalue,
                target_action_value=target_qvalue,
                policy_optimizer=policy_opt,
                action_value_optimizer=qvalue_opt,
                update_policy=(update % 2 == 0),
                device=device,
            )
            policy.std *= args.std_decay
        stats.update(update_stats)
        stats['update/iteration'] = iteration

        # evaluate policy
        if iteration % args.evaluation_frequency == 0:
            eval_env.reset()
            eval_replay = eval_env.run(
                lambda s: policy.act(s, deterministic=True),
                episodes=args.evaluation_episodes,
            )
            eval_rewards = eval_replay.reward().sum().item()
            stats['eval/rewards'] = eval_rewards / args.evaluation_episodes
            stats['eval/iteration'] = iteration

        # log to wandb
        wandb.log(stats)


if __name__ == "__main__":

    class OffPolicyArguments:

        env: str = 'HalfCheetah-v3'
        #  env: str = 'Ant-v3'
        algorithm: str = 'td3'
        num_iterations: int = 1000000
        warmup_steps: int = 10000
        batch_size: int = 128
        std_decay: float = 0.99997
        update_interval: int = 50
        learning_rate: float = 1e-3
        evaluation_frequency: int = 100
        evaluation_episodes: int = 3
        cuda: bool = False
        seed: int = 1234

    main(OffPolicyArguments())
