
.PHONY: all tests

all:
	python examples/actor_critic_pendulum.py

reinforce:
	python examples/reinforce_cartpole.py

ac:
	python examples/actor_critic_cartpole.py

tests:
	python -m unittest discover -s 'tests' -p '*_tests.py' -v
