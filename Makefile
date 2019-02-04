
.PHONY: all tests dist

all: dist

dist:
	mpirun -np 8 \
	       --oversubscribe \
	       -x OMP_NUM_THREADS=1 \
	       -x MKL_NUM_THREADS=1 \
	       python examples/dist_a2c_atari.py

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
