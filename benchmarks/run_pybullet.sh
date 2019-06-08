#!/usr/bin/env bash

# delayed twin SAC
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/pybullet/delayed_tsac_pybullet.py BENCH_ENV=CartPoleBulletEnv-v1    BENCH_SEED=42 python benchmark.py &
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/pybullet/delayed_tsac_pybullet.py BENCH_ENV=ReacherBulletEnv-v0     BENCH_SEED=42 python benchmark.py &
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/pybullet/delayed_tsac_pybullet.py BENCH_ENV=HalfCheetahBulletEnv-v0 BENCH_SEED=42 python benchmark.py &
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/pybullet/delayed_tsac_pybullet.py BENCH_ENV=MinitaurTrottingEnv-v0  BENCH_SEED=42 python benchmark.py &
wait

# PPO
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/pybullet/ppo_pybullet.py BENCH_ENV=CartPoleBulletEnv-v1    BENCH_SEED=42 python benchmark.py &
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/pybullet/ppo_pybullet.py BENCH_ENV=ReacherBulletEnv-v0     BENCH_SEED=42 python benchmark.py &
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/pybullet/ppo_pybullet.py BENCH_ENV=HalfCheetahBulletEnv-v0 BENCH_SEED=42 python benchmark.py &
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/pybullet/ppo_pybullet.py BENCH_ENV=MinitaurTrottingEnv-v0  BENCH_SEED=42 python benchmark.py &
wait

# SAC
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/pybullet/sac_pybullet.py BENCH_ENV=CartPoleBulletEnv-v1    BENCH_SEED=42 python benchmark.py &
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/pybullet/sac_pybullet.py BENCH_ENV=ReacherBulletEnv-v0     BENCH_SEED=42 python benchmark.py &
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/pybullet/sac_pybullet.py BENCH_ENV=HalfCheetahBulletEnv-v0 BENCH_SEED=42 python benchmark.py &
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BENCH_SCRIPT=../examples/pybullet/sac_pybullet.py BENCH_ENV=MinitaurTrottingEnv-v0  BENCH_SEED=42 python benchmark.py &
wait
