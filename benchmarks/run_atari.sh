#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=2 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/atari/ppo_atari.py BENCH_ENV=PongNoFrameskip-v4 BENCH_SEED=42 python benchmark.py

MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/atari/a2c_atari.py BENCH_ENV=PongNoFrameskip-v4     BENCH_SEED=42 python benchmark.py &
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/atari/a2c_atari.py BENCH_ENV=BreakoutNoFrameskip-v4 BENCH_SEED=42 python benchmark.py &
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/atari/a2c_atari.py BENCH_ENV=EnduroNoFrameskip-v4   BENCH_SEED=42 python benchmark.py &
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/atari/a2c_atari.py BENCH_ENV=SeaquestNoFrameskip-v4 BENCH_SEED=42 python benchmark.py &
wait


#MKL_NUM_THREADS=1 \
#OMP_NUM_THREADS=1 \
#BENCH_SCRIPT=../examples/atari/dist_a2c_atari.py \
#BENCH_ENV=PongNoFrameskip-v4 \
#BENCH_SEED=42 \
#python -m torch.distributed.launch --nproc_per_node=2 benchmark.py

#MKL_NUM_THREADS=1 \
#OMP_NUM_THREADS=1 \
#BENCH_SCRIPT=../examples/atari/dist_a2c_atari.py \
#BENCH_ENV=BreakoutNoFrameskip-v4 \
#BENCH_SEED=42 \
#python -m torch.distributed.launch --nproc_per_node=2 benchmark.py
