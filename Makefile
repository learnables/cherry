
.PHONY: all tests

THREAD_PER_PROC=4

all: dist

dist:
	MKL_NUM_THREADS=$(THREAD_PER_PROC) NUMEXPR_NUM_THREADS=$(THREAD_PER_PROC) OMP_NUM_THREADS=$(THREAD_PER_PROC) python examples/distributed_atari/main.py main --num_steps=10000000 --num_workers=16

acp:
	python examples/actor_critic_pendulum.py

reinforce:
	python examples/reinforce_cartpole.py

ac:
	python examples/actor_critic_cartpole.py

tests:
	python -m unittest discover -s 'tests' -p '*_tests.py' -v
