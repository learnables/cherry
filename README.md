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

# Development Guidelines

* The `master` branch is always working, considered stable.
* The `dev` branch should always work and is ahead of `master`, considered cutting edge.
* To implement a new functionality: branch `dev` into `your_name/functionality_name`, implement your functionality, then pull request to `dev`. It will be periodically merged into `master`.

# Usage

The following snippet demonstrates some of the tools offered by cherry.

~~~python
import cherry as ch

# Wrapping environments
env = ch.envs.Logger(env, interval=1000)  # Prints rollouts statistics
env = ch.envs.Normalized(env, normalize_state=True, normalize_reward=False)  
env = ch.envs.Torch(env)  # Converts actions/states to tensors

# Storing and retrieving experience
replay = ch.ExperienceReplay()
replay.add(old_state, action, reward, state, done, info = {
        'log_prob': mass.log_prob(action),  # Can add any variable/tensor to the transitions
        'value': value
})
replay.actions  # Tensor of all stored actions
replay.states  # Tensor of all stored states
replay.empty()  # Removes all stored experience

# Discounting and normalizing rewards
rewards = ch.rewards.discount_rewards(GAMMA, replay.rewards, replay.dones)
rewards = ch.utils.normalize(th.tensor(rewards))

# Sampling rollouts per episode or samples
num_samples, num_episodes = ch.rollouts.collect(env,
                                                get_action,
                                                replay,
                                                num_episodes=10,
                                                # alternatively: num_samples=1000,
)
~~~

Concrete examples are available in the [examples/](./examples/) folder.

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
* John Schulman's [implementations](https://github.com/joschu/modular_rl)
* Ilya Kostrikov's [implementations](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr),
* Shangtong Zhang's [implementations](https://github.com/ShangtongZhang/DeepRL),
* [RLLab](https://github.com/rll/rllab),
* [RLKit](https://github.com/vitchyr/rlkit).
