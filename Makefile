
.PHONY: all tests dist

THREAD_PER_PROC=1

all: dist

dist:
	python examples/distributed_atari/main.py main --num_steps=10000000 --env=PongNoFrameskip-v4

ppo:
	python examples/ppo_pybullet.py

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

publish:
	python setup.py sdist
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
