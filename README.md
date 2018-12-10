<p align="center"><img src="./docs/assets/img/cherry_full.png" height="150px" /></p>

--------------------------------------------------------------------------------

Cherry is reinforcement learning framework for researchers built on top of PyTorch.

Unlike other reinforcement learning implementations, cherry doesn't try to provide a single interface to existing algorithms.
Instead, it provides you with common tools to write your own algorithms.
Drawing from the UNIX philosophy, each tool strives to be as independent from the rest of the framework as possible.
So if you don't like a specific tool, you can still use parts of cherry without headaches.

# Installation

For now, cherry is still in development.

1. Clone the repo: `git clone https://github.com/seba-1511/cherry`
2. `cd cherry`
3. `pip install -e .`

Upon our first public release, you'll be able to

```
pip install cherry-rl
```

# Usage

The following snippet demonstrates some of the tools offered by cherry.

~~~python
import cherry as ch

def update(replay, optimizer):
    policy_loss = []
    value_loss = []

    # Discount and normalize rewards
    rewards = ch.rewards.discount_rewards(GAMMA, replay.list_rewards, replay.list_dones)
    rewards = ch.utils.normalize(th.tensor(rewards))

    # Compute losses
    for info, reward in zip(replay.list_infos, rewards):
        log_prob = info['log_prob']
        value = info['value']
        policy_loss.append(-log_prob * (reward - value.item()))
        value_loss.append(F.mse_loss(value, reward.detach()))

    # Take optimization step
    optimizer.zero_grad()
    loss = th.stack(policy_loss).sum() + V_WEIGHT * th.stack(value_loss).sum()
    loss.backward()
    optimizer.step()

env = gym.make('CartPole-v0')
env = ch.envs.Logger(env, interval=1000)
env = ch.envs.Normalized(env)
env = ch.envs.Torch(env)
env.seed(SEED)

policy = ActorCriticNet() # Standard nn.Module
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
replay = ch.ExperienceReplay()
get_action = lambda state: policy(state)

for episode in count(1):
    # We use the rollout collector, but could've written our own
    num_samples, num_episodes = ch.rollouts.collect(env,
                                                    get_action,
                                                    replay,
                                                    num_episodes=1)
    # Update policy the way we want
    update(replay, optimizer)
    replay.empty()
~~~

More fun stuff available in the [examples/](./examples/) folder.

# Documentation

The documentation will be written as we begin to converge the core concepts of cherry.

# TODO

Some functionalities that we might want to implement.

* normalize / serialize / parallelize environments,
* compute advantages / policy gradients / rewards,
* print and log training / debugging stats,
* distributed / async training,
* fix: what is up with the non-determinism ?
* sample batches from replay,
* function to sample experience,
* handle recurrent policies,
* functions for GAE, discounted and bootstrapped rewards,
* unified support for continuous and discrete environments,
* one high-performance implementation of A2C on Breakout and Ant-v1. (or pybullet equivalent)

# Acknowledgements

Cherry draws inspiration from many reinforcement learning implementations, including

* [OpenAI Baselines](https://github.com/openai/baselines),
* [RLLab](https://github.com/rll/rllab),
* Ilya Kostrikov's [implementations](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr),
* Shangtong Zhang's [implementations](https://github.com/ShangtongZhang/DeepRL),
* [RLKit](https://github.com/vitchyr/rlkit).
