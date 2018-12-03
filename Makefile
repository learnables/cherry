
.PHONY: all

all:
	python examples/reinforce_cartpole.py

reinforce:
	python examples/reinforce_cartpole.py

ac:
	python examples/actor_critic_cartpole.py
