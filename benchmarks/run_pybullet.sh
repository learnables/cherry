#!/usr/bin/env bash

MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py ../examples/pybullet/ppo_pybullet.py CartPoleBulletEnv-v0 42
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py ../examples/pybullet/ppo_pybullet.py RoboschoolAnt-v1 42
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py ../examples/pybullet/ppo_pybullet.py MinitaurTrottingEnv-v0 42
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py ../examples/pybullet/ppo_pybullet.py AntBulletEnv-v0 42
