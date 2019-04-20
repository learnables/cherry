#!/usr/bin/env bash

MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py ../examples/pybullet/ppo_pybullet.py MinitaurTrottingEnv-v0 42
