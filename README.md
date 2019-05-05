<p align="center"><img src="https://seba-1511.github.io/cherry/assets/img/cherry_full.png" height="150px" /></p>

--------------------------------------------------------------------------------

[![Test Status](https://travis-ci.com/seba-1511/cherry.svg?token=wS9Ya4UiannE2WzTjpHV&branch=master)](https://travis-ci.com/seba-1511/cherry)

Cherry is a reinforcement learning framework for researchers built on top of PyTorch.

Unlike other reinforcement learning implementations, cherry doesn't implement a single monolithic  interface to existing algorithms.
Instead, it provides you with low-level, common tools to write your own algorithms.
Drawing from the UNIX philosophy, each tool strives to be as independent from the rest of the framework as possible.
So if you don't like a specific tool, you don’t need to use it.

**Features**

* Pythonic and modular interface *à la* Pytorch.
* Support for tabular (!) and function approximation algorithms.
* Various OpenAI Gym environment wrappers.
* Helper functions for popular algorithms. (e.g. A2C, DDPG, TRPO, PPO, SAC)
* Logging, visualization, and debugging tools.
* Painless and efficient distributed training on CPUs and GPUs.
* Unit, integration, and regression tested, continuously integrated.

To learn more about the tools and philosophy behind cherry, check out our [Getting Started tutorial](http://cherry-rl.net/tutorials/getting_started/).

## Example

The following snippet showcases some of the tools offered by cherry.

~~~python
import cherry as ch

# Wrapping environments
env = ch.envs.Logger(env, interval=1000)  # Prints rollouts statistics
env = ch.envs.Normalized(env, normalize_state=True, normalize_reward=False)  
env = ch.envs.Torch(env)  # Converts actions/states to tensors

# Storing and retrieving experience
replay = ch.ExperienceReplay()
replay.append(old_state,
              action,
              reward,
              state,
              done,
              log_prob=mass.log_prob(action),  # Can add any variable/tensor to the transitions
              value=value
            )
replay.action()  # Tensor of all stored actions
replay.state()  # Tensor of all stored states
replay.empty()  # Removes all stored experience

# Discounting and normalizing rewards
rewards = ch.td.discount(GAMMA, replay.rewards, replay.dones)
rewards = ch.normalize(rewards)

# Sampling rollouts per episode or samples
env = envs.Runner(env)
replay = env.run(get_action, steps=100)  # alternatively: episodes=10

~~~

Many more high-quality examples are available in the [examples/](./examples/) folder.

## Installation

**Note** Cherry is considered in early alpha release. Stuff might break.

```
pip install cherry-rl
```

## Documentation

Documentation and tutorials are available on cherry’s website: [http://cherry-rl.net](http://cherry-rl.net).

## Contributing

First, thanks for your consideration in contributing to cherry.
Here are a couple of guidelines we strive to follow.

* It's always a good idea to open an issue first, where we can discuss how to best proceed.
* Branch/fork from `dev`, and create a pull request as soon as possible to allow for early discussions.
* If you want to contribute a new example using cherry, it would preferably stand in a single file.
* If you would like to contribute a new feature to the core library, we suggest to first implement an example showcasing your new functionality. Doing so is quite useful:
    * it allows for automatic testing,
    * it ensures that the functionality is correctly implemented,
    * it shows users how to use your functionality, and
    * it gives a concrete example when discussing the best way to merge your implementation.

We don't have forums, but are happy to discuss with you on slack.
Make sure to send an email to [smr.arnold@gmail.com](smr.arnold@gmail.com) to get an invite.

## Acknowledgements

Cherry draws inspiration from many reinforcement learning implementations, including

* [OpenAI Baselines](https://github.com/openai/baselines),
* John Schulman's [implementations](https://github.com/joschu/modular_rl)
* Ilya Kostrikov's [implementations](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr),
* Shangtong Zhang's [implementations](https://github.com/ShangtongZhang/DeepRL),
* Dave Abel's [implementations](https://github.com/david-abel/simple_rl/),
* Vitchyr Pong's [implementations](https://github.com/vitchyr/rlkit),
* Kai Arulkumaran's [implementations](https://github.com/Kaixhin/spinning-up-basic),
* [RLLab](https://github.com/rll/rllab).


## Why 'cherry' ?

Because it's the sweetest part of [the cake](https://twitter.com/ylecun/status/1097532314614034433).
