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

