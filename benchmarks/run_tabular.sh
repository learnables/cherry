#!/usr/bin/env bash

# Q-Learning
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/tabular/q_learning.py BENCH_ENV=CliffWalking-v0 BENCH_SEED=42 python benchmark.py &
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/tabular/q_learning.py BENCH_ENV=FrozenLake-v0   BENCH_SEED=42 python benchmark.py &
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/tabular/q_learning.py BENCH_ENV=NChain-v0       BENCH_SEED=42 python benchmark.py &
wait

# SARSA
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/tabular/sarsa.py BENCH_ENV=CliffWalking-v0 BENCH_SEED=42 python benchmark.py &
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/tabular/sarsa.py BENCH_ENV=FrozenLake-v0   BENCH_SEED=42 python benchmark.py &
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/tabular/sarsa.py BENCH_ENV=NChain-v0       BENCH_SEED=42 python benchmark.py &
wait
