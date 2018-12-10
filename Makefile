
.PHONY: all tests

all: dist

dist:
	python examples/distributed_atari/main.py main --num_steps=100 --num_workers=1

acp:
	python examples/actor_critic_pendulum.py

reinforce:
	python examples/reinforce_cartpole.py

ac:
	python examples/actor_critic_cartpole.py

tests:
	python -m unittest discover -s 'tests' -p '*_tests.py' -v
