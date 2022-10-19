<p align="center"><img src="http://cherry-rl.net/assets/images/cherry_full.png" height="150px" /></p>

--------------------------------------------------------------------------------

[![Build Status](https://travis-ci.org/learnables/cherry.svg?branch=master)](https://travis-ci.org/learnables/cherry)

Cherry is a reinforcement learning framework for researchers built on top of PyTorch.

Unlike other reinforcement learning implementations, cherry doesn't implement a single monolithic  interface to existing algorithms.
Instead, it provides you with low-level, common tools to write your own algorithms.
Drawing from the UNIX philosophy, each tool strives to be as independent from the rest of the framework as possible.
So if you don't like a specific tool, you don’t need to use it.

**Features**

* Pythonic and low-level interface *à la* Pytorch.
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

# Wrap environments
env = gym.make('CartPole-v0')
env = ch.envs.Logger(env, interval=1000)
env = ch.envs.Torch(env)

policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
replay = ch.ExperienceReplay()  # Manage transitions

for step in range(1000):
    state = env.reset()
    while True:
        mass = Categorical(policy(state))
        action = mass.sample()
        log_prob = mass.log_prob(action)
        next_state, reward, done, _ = env.step(action)

        # Build the ExperienceReplay
        replay.append(state, action, reward, next_state, done, log_prob=log_prob)
        if done:
            break
        else:
            state = next_state

    # Discounting and normalizing rewards
    rewards = ch.td.discount(0.99, replay.reward(), replay.done())
    rewards = ch.normalize(rewards)

    loss = -th.sum(replay.log_prob() * rewards)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    replay.empty()
~~~

Many more high-quality examples are available in the [examples/](./examples/) folder.

## Installation

**Note** Cherry is considered in early alpha release. Stuff might break.

```
pip install cherry-rl
```

## Changelog

A human-readable changelog is available in the [CHANGELOG.md](CHANGELOG.md) file.

## Documentation

Documentation and tutorials are available on cherry’s website: [http://cherry-rl.net](http://cherry-rl.net).

## Contributing

First, thanks for your consideration in contributing to cherry.
Here are a couple of guidelines we strive to follow.

* It's always a good idea to open an issue first, where we can discuss how to best proceed.
* If you want to contribute a new example using cherry, it would preferably stand in a single file.
* If you would like to contribute a new feature to the core library, we suggest to first implement an example showcasing your new functionality. Doing so is quite useful:
    * it allows for automatic testing,
    * it ensures that the functionality is correctly implemented,
    * it shows users how to use your functionality, and
    * it gives a concrete example when discussing the best way to merge your implementation.

We don't have forums, but are happy to discuss with you on slack.
Make sure to send an email to [smr.arnold@gmail.com](mailto:smr.arnold@gmail.com) to get an invite.

## Acknowledgements

Cherry draws inspiration from many reinforcement learning implementations, including

* [OpenAI Baselines](https://github.com/openai/baselines),
* John Schulman's [implementations](https://github.com/joschu/modular_rl)
* Ilya Kostrikov's [implementations](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr),
* Shangtong Zhang's [implementations](https://github.com/ShangtongZhang/DeepRL),
* Dave Abel's [implementations](https://github.com/david-abel/simple_rl/),
* Vitchyr Pong's [implementations](https://github.com/vitchyr/rlkit),
* Kai Arulkumaran's [implementations](https://github.com/Kaixhin/spinning-up-basic),
* [RLLab](https://github.com/rll/rllab) / [Garage](https://github.com/rlworkgroup/garage).


## Why 'cherry' ?

Because it's the sweetest part of [the cake](https://twitter.com/ylecun/status/1097532314614034433).
