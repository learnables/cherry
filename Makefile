
.PHONY: all tests dist

THREAD_PER_PROC=1

all: ppo

dist:
	mpirun -n 8 python examples/distributed_atari/main.py main --num_steps=10000000 --env=PongNoFrameskip-v4

ppo:
	python examples/ppo_pendulum.py

acp:
	python examples/actor_critic_pendulum.py

reinforce:
	python examples/reinforce_cartpole.py

ac:
	python examples/actor_critic_cartpole.py

grid:
	python examples/actor_critic_gridworld.py

tests:
	python -m unittest discover -s 'tests' -p '*_tests.py' -v
