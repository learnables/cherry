<p align="center"><img src="http://cherry-rl.net/assets/images/cherry_full.png" height="150px" /></p>

--------------------------------------------------------------------------------

[![Build Status](https://travis-ci.org/learnables/cherry.svg?branch=master)](https://travis-ci.org/learnables/cherry)

Cherry is a reinforcement learning framework for researchers built on top of PyTorch.

Unlike other reinforcement learning implementations, cherry doesn't implement a single monolithic  interface to existing algorithms.
Instead, it provides you with low-level, common tools to write your own algorithms.
Drawing from the UNIX philosophy, each tool strives to be as independent from the rest of the framework as possible.
So if you don't like a specific tool, you don’t need to use it.

**Features**

Cherry extends PyTorch with only a handful of new core concepts.

* PyTorch Modules for reinforcement learning: 
    * `cherry.nn.Policy`:
    * `cherry.nn.ActionValue`:
    * `cherry.nn.StateValue`: 
* Data structures for reinforcement learning:
    * `cherry.Transition`:
    * `cherry.ExperienceReplay`: 

Cherry also includes additional features, to help implement existing and new RL algorithms.

* Pythonic and low-level interface *à la* Pytorch.
* Support for tabular (!) and function approximation algorithms.
* Various OpenAI Gym environment wrappers.
* Helper functions for popular algorithms. (e.g. A2C, DDPG, TRPO, PPO, SAC)
* Logging, visualization, and debugging tools.
* Painless and efficient distributed training on CPUs and GPUs.
* Unit, integration, and regression tested, continuously integrated.

To learn more about the tools and philosophy behind cherry, check out our [Getting Started tutorial](http://cherry-rl.net/tutorials/getting_started/).

## Overview and Examples

The following snippet showcases a few of the tools offered by cherry.
Many more high-quality examples are available in the [examples/](./examples/) folder.


<details>
<summary><code>Using cherry.nn.Policy</summary>code></summary>

~~~python
class VisionPolicy(cherry.nn.Policy):  # inherits from torch.nn.Module
   
   def __init__(self, feature_extractor, actor):
      super(VisionGaussianPolicy, self).__init__()
      self.feature_extractor = feature_extractor
      self.actor = actor

   def forward(self, obs):
      mean = self.actor(self.feature_extractor(obs))
      std = 0.1 * torch.ones_like(mean)
      return cherry.distributions.TanhNormal(mean, std)  # policies always return a distribution

policy = VisionPolicy(MyResnetExtractor(), MyMLPActor())
dist = policy(obs)
action = policy.act(obs)  # sampled from policy's distribution
deterministic_action = policy.act(obs, deterministic=True)  # distribution's mode
~~~
</details>

## Installation

```
pip install cherry-rl
```

## Changelog

A human-readable changelog is available in the [CHANGELOG.md](CHANGELOG.md) file.

## Documentation

Documentation and tutorials are available on cherry’s website: [http://cherry-rl.net](http://cherry-rl.net).

## Contributing

Here are a couple of guidelines we strive to follow.

* It's always a good idea to open an issue first, where we can discuss how to best proceed.
* If you want to contribute a new example using cherry, it would preferably stand in a single file.
* If you would like to contribute a new feature to the core library, we suggest to first implement an example showcasing your new functionality. Doing so is quite useful:
    * it allows for automatic testing,
    * it ensures that the functionality is correctly implemented,
    * it shows users how to use your functionality, and
    * it gives a concrete example when discussing the best way to merge your implementation.

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
