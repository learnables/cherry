#!/usr/bin/env bash

# Q-Learning
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py ../examples/tabular/q_learning.py CliffWalking-v0 42
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py ../examples/tabular/q_learning.py FrozenLake-v0 42
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py ../examples/tabular/q_learning.py NChain-v0 42

# SARSA
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py ../examples/tabular/sarsa.py CliffWalking-v0 42
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py ../examples/tabular/sarsa.py FrozenLake-v0 42
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py ../examples/tabular/sarsa.py NChain-v0 42
