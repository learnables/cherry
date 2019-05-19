#!/usr/bin/env bash

MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/spinning-up/cherry_vpg.py  BENCH_ENV=Pendulum-v0 BENCH_SEED=42 python benchmark.py &
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/spinning-up/cherry_ddpg.py BENCH_ENV=Pendulum-v0 BENCH_SEED=42 python benchmark.py &
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/spinning-up/cherry_ppo.py  BENCH_ENV=Pendulum-v0 BENCH_SEED=42 python benchmark.py &
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/spinning-up/cherry_sac.py  BENCH_ENV=Pendulum-v0 BENCH_SEED=42 python benchmark.py &
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/spinning-up/cherry_dqn.py  BENCH_ENV=Pendulum-v0 BENCH_SEED=42 python benchmark.py &
wait
