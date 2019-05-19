
# Benchmarks

## Description

The benchmarking script `benchmark.py` allows to run some implementations for various environments, while logging the training on [Weights and Biases](https://wandb.ai/), saving the trained model weights, and rendering visualizations.

The results are publicly available here: [Weights & Biases Reports](https://app.wandb.ai/arnolds/cherry-benchmarks)

The script reads the source code of an implementation and runs some specific (pre-defined) functions that will execute training.
It's a dirty hack, so use it at your own peril :)

## Install and Run

To install:

0. `cd benchmarks`
1. `pip install -r requirements.txt`
2. `wandb login YOUR_WANDB_KEY`

To run, from the `benchmarks` directory:

* `./run_tabular.sh` (~ 5 min)
* `./run_spinup.sh` (~ 50 min)
* `./run_pybullet.sh` (~ 500 min)
* `./run_atari.sh` (~ 5000 min)

## Wrapping your own scripts

To wrap your own scripts, (e.g. if they are to be contributed to cherry's `examples/` directory) they will have to implement a `main(env='Name')` function.
Here `Name` is the name of the environment to use, for example `CartPole-v0`.
(Note: no variable other than `env` will be instantiated.)
Assuming your script is named `myscript.py`, `benchmark.py` will read that file and call `exec()` on its content.
Then, it will parse the code inside `main` and run it.
Finally, it will try to fetch different variables based on the pre-defined heuristics listed below.

To benchmark your script, call

```
BENCH_SCRIPT=path/to/myscript.py BENCH_ENV=EnvironmentName-v0 BENCH_SEED=42 python benchmark.py --myarg=1234
```

where `--myarg=1234` is an argument to be parsed by your script.

### Pre-defined heuristics

**Using cherry's Logger**
If your script uses `cherry.envs.Logger`, or one of its subclass, `benchmark.py` will automatically log all logged values to W&B.
It does so by wrapping the method `Logger.log` so that it first calls `wandb.log` and then the original `log` function, so the logging happens in real-time.

Using `Logger` will also enable to report the mean episode length, mean reward over past 10 episodes, all obtained rewards, and sum of rewards per episode.

**SEED global variable**
`benchmark.py` will automatically seed `random`, `numpy`, and `torch`.
In addition, if your script defines a global variable `SEED` it will overwrite it with the value of `BENCH_SEED`.
This is useful if you need to custom seed your environment, as is the case for the distributed examples.

**Saving weights**
If your `main` function or your script defines a variable `model`, `policy`, `agent`, `actor`, or `critic` and that variable has a method `state_dict` available, the output of `state_dict` will be saved in the 

**Testing Results**
Once main has completed if `env` has a method `run` and either `get_action` or `agent` is defined, `benchmark.py` will try to run the 25 episodes of the environment and report the sum of rewards for each episode.

**Visualization**
TODO: Save 5 visualizations of a rollout post-training.
