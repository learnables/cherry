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
For example, the `OpenAIAtari` wrapper requires OpenCV (`pip install opencv-python`) and the `VisdomLogger` requires visdom (`pip install visdom`).

**Note**
While cherry depends on `gym` for its environment wrappers, it doesn't restrict you to Gym environemnts.
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

#### Transitions and Experience Replay

#### Temporal Difference and Policy Gradients

#### Models and PyTorch

#### Gym Wrappers

#### Plots

Reporting comparable results has become a central problem in modern reinforcement learning.
In order to alleviate this issue, cherry provides utilities to smooth and compute confidence intervals over lists of rewards.
Those are available in the [cherry.plot](http://cherry-rl.net/docs/cherry.plot/) submodule.

## Implementing REINFORCE

**Note**
One of the characteristics of cherry is to avoid providing "pre-baked" algorithms.
Having a look at the numerous [examples](https://github.com/seba-1511/cherry/tree/master/examples) might be a good source of inspiration.

