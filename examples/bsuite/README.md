# Behaviour Suite for Reinforcement Learning

The examples in this folder demonstrate how to interface cherry with DeepMind's [Behaviour Suite for Reinforcement Learning](https://github.com/deepmind/bsuite) (bsuite).

## Requirements

* cherry: `pip install cherry-rl`
* bsuite: `pip install git+git://github.com/deepmind/bsuite.git`
* tqdm: `pip install tqdm`

## Examples Descriptions

### trpo_v_random.py

**Status** Working.

Generates results from a Random and TRPO agent on the whole sweep of tests.
Shows how to generate general plots (radar, comparison) as well as specific analysis (bandits).
