<p align="center"><img src="./docs/assets/img/cherry_full.png" height="150px" /></p>

--------------------------------------------------------------------------------

Cherry is reinforcement learning framework for researchers built on top of PyTorch.

Unlike other reinforcement learning implementations, cherry doesn't 

# Installation

For now, cherry is still in development.

1. Clone the repo: `git clone https://github.com/seba-1511/cherry`
2. `cd cherry`
3. `pip install -e cherry`

Upon our first public release, you'll be able to

```
pip install cherry
```

# Usage

The following snippet demonstrates some of the tools offered by cherry.

~~~python
env = gym.make('CartPole-v0')
env = envs.Logger(env, interval=1000)
env = envs.Normalized(env)
env = envs.Torch(env)
env.seed(SEED)

policy = ActorCriticNet()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
running_reward = 10.0
replay = ch.ExperienceReplay()
get_action = lambda state: get_action_value(state, policy)

for episode in count(1):
    # We use the rollout collector, but could've written our own
    num_samples, num_episodes = rollouts.collect(env,
                                                 get_action,
                                                 replay,
                                                 num_episodes=1)
    # Update policy
    update(replay, optimizer)
    replay.empty()
~~~

More fun stuff available in the [examples/](./examples/) folder.

# Documentation

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

Cherry draws inspiration from many reinforcement learning implementations.

* [OpenAI Baselines](https://github.com/openai/baselines)
* [RLLab](https://github.com/rll/rllab)
* Ilya Kostrikov's [implementations](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)
* Shangtong Zhang's [implementations](https://github.com/ShangtongZhang/DeepRL)
* [RLKit](https://github.com/vitchyr/rlkit)
