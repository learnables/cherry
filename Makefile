
.PHONY: all tests dist

THREAD_PER_PROC=8

all: dist

dist:
	mpirun -n 2 -x MKL_NUM_THREADS=$(THREAD_PER_PROC) python examples/distributed_atari/main.py main --num_steps=10000000

ppo:
	python examples/ppo_pybullet.py

acp:
	python examples/actor_critic_pendulum.py

reinforce:
	python examples/reinforce_cartpole.py

ac:
	python examples/actor_critic_cartpole.py

tests:
	python -m unittest discover -s 'tests' -p '*_tests.py' -v
