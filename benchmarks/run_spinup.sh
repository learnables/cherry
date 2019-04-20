#!/usr/bin/env bash

MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py ../examples/spinning-up/cherry_vpg.py Pendulum-v0 42
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py ../examples/spinning-up/cherry_ppo.py Pendulum-v0 42
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py ../examples/spinning-up/cherry_ddpg.py Pendulum-v0 42
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py ../examples/spinning-up/cherry_sac.py Pendulum-v0 42
