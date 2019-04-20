#!/usr/bin/env bash

# To make atari run, we need to pass the benchmark arguments as environment variables.
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 benchmark.py ../examples/atari/dist_a2c_atari.py PongNoFrameskip-v4 42

#MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=16 benchmark.py ../examples/pybullet/dist_a2c_atari.py BreakoutNoFrameskip-v4 42
