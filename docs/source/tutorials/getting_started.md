# Getting Started with Cherry

This document provides an overview of the philosophy behind cherry, the tools it provides, and a small illustrative example.
By the end of this tutorial, you should be well-equiped to incorporate cherry in your research workflow.

We assume that you are already familiar with reinforcement learning.
If, instead, you're looking for an introduction to the field we recommend looking at Josh Achiam's [Spinning Up in Deep RL](http://spinningup.openai.com/).

## Installation

The first step in getting started with cherry is to install it.
Since it is implemented purely on top of PyTorch, you can easily do that by typing the following command in your favorite shell.

```shell
pip install cherry-rl
```

By default cherry only has two dependencies: `torch` and `gym`.
However, more dependencies might be required if you plan to use some more specific functionalities.
For example, the [OpenAIAtari]() wrapper requires OpenCV (`pip install opencv-python`) and the [VisdomLogger]() requires visdom (`pip install visdom`).

**Note**
While cherry depends on Gym for its environment wrappers, it doesn't restrict you to Gym environments.
For instance, check the examples using [simple_rl](https://github.com/seba-1511/cherry/tree/master/examples/simple_rl) and [pycolab](https://github.com/seba-1511/cherry/tree/master/examples/pycolab) environments for Gym-free usage of cherry.

## Overview

**Why do we need cherry ?**
There are many reinforcement learning libraries, many of which feature high-quality implementations.
However, few of them provide the kind of low-level utilities useful to researchers.
Cherry aims to alleviate this issue.
Instead of an interface akin to `PPO(env_name).train(1000)`, it provides researchers with a set of tools they can use to write readable, replicable, and flexible implementations.
Cherry prioritizes time-to-correct-implementation over time-to-run, by explicitely helping you check, debug, and reliably report your results.


**How to use cherry ?**
Our goal is to make cherry a natural extension to PyTorch with reinforcement learning in mind.
To this end, we closely follow the package structure of PyTorch while providing additional utilities where we saw fit.
So if your goal is to implement a novel distributed off-policy policy gradient algorithm, you can count on cherry to provide you experience replays, policy gradient losses, discounting and advantage functions, and distributed optimizers.
Those functions not only reduce the time spent writing code, they also check that your implementation is sane.
(e.g. do the log probabilities and rewards have identical shapes?)
Moreover, cherry provides implementation details -- unfortunately -- necessary to make deep reinforcement learning work.
(e.g. initializations, modules, and wrappers commonly used in robotic or Atari benchmarks.)
Importantly, it includes built-in debugging functionalities: cherry can help you visualize what is happening under the hood of your algorithm to help you find bugs faster.


**What's the future of cherry ?**
Reinforcement learning is a fast moving field, and it is difficult to predict which advances are safe bets for the future.
Our long-term development strategy can be summarized as follows.

1. Have as many recent and high-quality [examples](https://github.com/seba-1511/cherry/tree/master/examples) as possible.
2. Merge advances that turn up to be fundamental in theory or practice into the core library.

We hope to combat the reproducibility crisis by extensively [testing](https://github.com/seba-1511/cherry/tree/master/tests) and [benchmarking](https://github.com/seba-1511/cherry/tree/master/benchmarks) our implementations.

**Note** Cherry is in its early days and is still missing some of the well-established methods from the past 60 years.
Those ones are being implemented as fast as we can :)

## Core Features

The following features are fundamental components of cherry.

#### Transitions and Experience Replay

A majority of algorithms needs to store, retrieve, and sample past experience.
To do that, you can use cherry's [ExperienceReplay]().
An experience replay is implemented as a wrapper around a standard Python list.
The major difference is that the [append()]() method expects arguments used to create a [Transition]().
In addition to behaving like a list, it exposes methods that act on this list, such as [to(device)]() (moves the replay to a device), [sample()]() (randomly samples some experience), or [load()]()/[save()]() (for convenient serialization).
An [ExperienceReplay]() contains [Transition]()s, which are akin to (`state`, `action`, `reward`, `next_state`, `done`) named tuples with possibly additional custom fields.
Those fields are easily accessible directly from the replay by accessing the method named after them.
For example, calling `replay.action()` will fetch the action field from every transition stored in `replay`, stack them along the first dimension, and return that large tensor.
The same is true for custom fields; if all transitions have a `logprob` field, `replay.logprob()` will return the result of stacking them.

#### Temporal Difference and Policy Gradients

Many low-level utilities used to implement temporal difference and policy gradient algorithms are available in the [cherry.td]() and [cherry.pg]() modules, respectively.
Those modules include classical methods such as [discounting rewards]() or computing the [temporal difference](), as well as more recent advances such as the [generalized advantage estimator]().
We tried our best to avoid philosophical dissonance when a method belonged to both families of algorithms, but it is our understanding that conflict is innevitable.

#### Models and PyTorch

Similar to PyTorch, we provide differentiable modules in [cherry.nn](), domain-specific initialization schemes in [cherry.nn.init](), and optimization utilities in [cherry.optim]().
In addition, popular higher-level models are available in [cherry.models](); for instance, those include [tabular modules](), the Atari CNN [features extractor](), and a [Multi-Layer Perceptron]() for continuous control.

#### Gym Wrappers

Given the popularity of OpenAI Gym environment in modern reinforcement learning benchmarks, cherry includes convenient wrappers in the [cherry.envs]() package.
Examples include normalization of [states]() and [actions](), [Atari]() frames pre-processing, customizable [action]() / [reward]() processing, and automatic collection of [experience]() in a replay.

#### Plots

Reporting comparable results has become a central problem in modern reinforcement learning.
In order to alleviate this issue, cherry provides utilities to smooth and compute confidence intervals over lists of rewards.
Those are available in the [cherry.plot](http://cherry-rl.net/docs/cherry.plot/) submodule.

## Implementing Policy Gradient

The following snippet demonstrates how to use cherry to implement the policy gradient theorem.

~~~python
import cherry as ch

env = gym.make('CartPole-v0')
env = ch.envs.Logger(env, interval=1000)
env = ch.envs.Torch(env)
env = ch.envs.Runner(env)
env.seed(42)

policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
action_dist = ch.distributions.ActionDistribution(env)

def get_action(state):
    mass = action_dist(policy(state))
    action = mass.sample()
    log_prob = mass.log_prob(action)
    return action, {'log_prob': log_prob}

for step in range(1000):
    replay = env.run(get_action, episodes=1)

    rewards = ch.td.discount(0.99, replay.reward(), replay.done())
    rewards = ch.normalize(rewards)

    loss = -th.sum(replay.log_prob() * rewards)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
~~~

After importing cherry, the first step is to instanciate, wrap, and seed the desired gym environment.

~~~python
env = gym.make('CartPole-v0')
env = ch.envs.Logger(env, interval=1000)
env = ch.envs.Torch(env)
env = ch.envs.Runner(env)
env.seed(42)
~~~

The [Logger](), [Torch](), and [Runner]() classes are Gym [environment wrappers]() that systematically modify the behaviour of an environment:

* `Logger` keeps track of metrics and prints them at a given interval.
* `Torch` converts Gym states into PyTorch tensors, and action tensors into numpy arrays.
* `Runner` implements a `run` method which allows to easily gather transitions for a number of steps or episodes.

One particularity of wrappers is that they automatically expose the methods of the wrapped environment if it is not reimplemented: so when calling `env.seed(42)` we are calling the method from `CartPole-v0`.

Second, we instanciate the policy, optimizer, as well as the action distribution.
The action distribution is created by calling

~~~python
action_dist = ch.distributions.ActionDistribution(env)
~~~

which will automatically choose between a diagonal Gaussian for continuous action-spaces or a categorical distribution discrete ones.

Next, we define `get_action` which defines how to get an action from our agent and will be used in conjuction to `env.run()` to quickly collect experience data:

~~~python
replay = env.run(get_action, episodes=1)
~~~

`env.run()` assumes that the first returned value by `get_action` is the action to be passed to the environment and the second, optional, returned value is a dictionary to be saved into the experience replay.
Under the hood, `env.run()` creates a new [ExperienceReplay]() and fills it with the desired number of transitions;
instead of `episodes=1` we could have passed `steps=100`.

Finally, we discount and normalize the rewards and take an optimization step on the policy gradient loss.

~~~python
rewards = ch.td.discount(0.99, replay.reward(), replay.done())
rewards = ch.normalize(rewards)

loss = -th.sum(replay.log_prob() * rewards)
optimizer.zero_grad()
loss.backward()
optimizer.step()
~~~

When calling `replay.reward()`, `replay.done()`, or `replay.log_prob()`, the experience replay will concatenate the corresponding attribute across all of its transitions and **return a new tensor**.
This means that this operation is rather expensive (cache it when possible) and that modifying this tensor does not modify the corresponding transitions in `replay`.
Note that in this case `log_prob` is a custom attribute which is not declared in the original implementation of `ExperienceReplay`, and we could have given it any name by changing the dictionary key in `get_action()`.


### Conclusion
You should now be able to use cherry in your own work.
For more information, have a look at the [documentation](http://cherry-rl.net/docs/cherry/), the other [tutorials](http://cherry-rl.net/tutorials/distributed_ppo/), or the numerous [examples](https://github.com/seba-1511/cherry/tree/master/examples). 
Since one of the characteristics of cherry is to avoid providing "pre-baked" algorithms, we tried our best to heavily document its usage.
