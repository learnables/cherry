<p align="center"><img src="http://cherry-rl.net/assets/images/cherry_full.png" height="128px" /></p>

--------------------------------------------------------------------------------

[![Build Status](https://travis-ci.org/learnables/cherry.svg?branch=master)](https://travis-ci.org/learnables/cherry)

Cherry is a reinforcement learning framework for researchers built on top of PyTorch.

Unlike other reinforcement learning implementations, cherry doesn't implement a single monolithic  interface to existing algorithms.
Instead, it provides you with low-level, common tools to write your own algorithms.
Drawing from the UNIX philosophy, each tool strives to be as independent from the rest of the framework as possible.
So if you don't like a specific tool, you don’t need to use it.

**Features**

Cherry extends PyTorch with only a handful of new core concepts.

* PyTorch modules for reinforcement learning: 
    * [`cherry.nn.Policy`](http://cherry-rl.net/api/cherry.nn/#cherry.nn.policy.Policy): base class for $\pi(a \mid s)$ policies.
    * [`cherry.nn.ActionValue`](http://cherry-rl.net/api/cherry.nn/#cherry.nn.action_value.ActionValue): base class for $Q(s, a)$ action-value functions.
* Data structures for reinforcement learning compatible with PyTorch:
    * [`cherry.Transition`](http://cherry-rl.net/api/cherry/#cherry.experience_replay.Transition): namedtuple to store $(s_t, a_t, r_t, s_{t+1})$ transitions (and more).
    * [`cherry.ExperienceReplay`](http://cherry-rl.net/api/cherry/#cherry.experience_replay.ExperienceReplay): a list-like buffer to store and sample transitions.
 * Low-level interface *à la* PyTorch to write and debug your algorithms.
    * [`cherry.td.*`](http://cherry-rl.net/api/cherry.td/) and [`cherry.pg.*`](http://cherry-rl.net/api/cherry.pg/): temporal difference and policy gradient utilities.
    * [`cherry.algorithms.*`](http://cherry-rl.net/api/cherry.algorithms/): helper functions for popular algorithms ([PPO](http://cherry-rl.net/api/cherry.algorithms/#cherry.algorithms.ppo.PPO), [TD3](http://cherry-rl.net/api/cherry.algorithms/#cherry.algorithms.td3.TD3), [DrQ](http://cherry-rl.net/api/cherry.algorithms/#cherry.algorithms.drq.DrQ), and [more](http://cherry-rl.net/api/cherry.algorithms/)).
    * [`cherry.debug.*`](http://cherry-rl.net/api/cherry.debug/) and [`cherry.plot.*`](http://cherry-rl.net/api/cherry.plot/): logging, visualization, and debugging tools.

To learn more about the tools and philosophy behind cherry, check out our [Getting Started tutorial](http://cherry-rl.net/tutorials/getting_started/).

## Overview and Examples

The following snippet showcases a few of the tools offered by cherry.
Many more high-quality examples are available in the [examples/](./examples/) folder.

#### Defining a [`cherry.nn.Policy`](http://cherry-rl.net/api/cherry.nn/#cherry.nn.policy.Policy)

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
action = policy.act(obs)  # sampled from policy's distribution
deterministic_action = policy.act(obs, deterministic=True)  # distribution's mode
action_distribution = policy(obs)  # work with the policy's distribution
~~~

#### Building a [`cherry.ExperienceReplay`](http://cherry-rl.net/api/cherry/#cherry.experience_replay.ExperienceReplay) of [`cherry.Transition`](http://cherry-rl.net/api/cherry/#cherry.experience_replay.Transition)

~~~python
# building the replay
replay = cherry.ExperienceReplay()
state = env.reset()
for t in range(1000):
   action = policy.act(state)
   next_state, reward, done, info = env.step(action)
   replay.append(state, action, reward, next_state, done)
   next_state = state

# manipulating the replay
replay = replay[-256:]  # indexes like a list
batch = replay.sample(32, contiguous=True)  # sample transitions into a replay
batch = batch.to('cuda') # move replay to device
for transition in reversed(batch): # iterate over a replay
   transition.reward *= 0.99

# get all states, actions, and rewards as PyTorch tensors.
reinforce_loss = - torch.sum(policy(batch.state()).log_prob(batch.action()) * batch.reward())
~~~

#### Designing algorithms with [`cherry.td`](http://cherry-rl.net/api/cherry.td/), [`cherry.pg`](http://cherry-rl.net/api/cherry.pg/), and [`cherry.algorithms`](http://cherry-rl.net/api/cherry.algorithms/)

~~~python
# defining a new algorithm
@dataclasses.dataclass
class MyA2C:
   discount: float = 0.99
   
   def update(self, replay, policy, state_value, optimizer):
      # discount rewards
      values = state_value(replay.action())
      discounted_rewards = cherry.td.discount(
         self.discount, replay.reward(), replay.done(), bootstrap=values[-1].detach()
      )

      # Compute losses
      policy_loss = cherry.algorithms.A2C.policy_loss(
         log_probs=policy(replay.state()).log_prob(replay.action()),
         advantages=discounted_rewards - values.detach(),
      )
      value_loss = cherry.algorithms.A2C.state_value_loss(values, discounted_rewards)

      # Optimization step
      optimizer.zero_grad()
      (policy_loss + value_loss).backward()
      optimizer.step()
      return {'a2c/policy_loss': policy_loss, 'a2c/value_loss': value_loss}

# using MyA2C
my_a2c = MyA2C(discount=0.95)
my_policy = MyPolicy()
linear_value = cherry.models.LinearValue(128)
adam = torch.optim.Adam(policy.parameters())
for step in range(1000):
   replay = collect_experience(policy)
   my_a2c.update(replay, my_policy, linear_value, adam)
~~~

## Install

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
