
.PHONY: all tests dist docs

all: sac

dist:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python -m torch.distributed.launch \
	          --nproc_per_node=8 \
		    examples/dist_a2c_atari.py

ppo:
	python examples/ppo_pybullet.py

acp:
	python examples/actor_critic_pendulum.py

reinforce:
	python examples/reinforce_cartpole.py

ac:
	python examples/actor_critic_cartpole.py

sac:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python examples/sac_pybullet.py

grid:
	python examples/actor_critic_gridworld.py

dqn:
	python examples/dqn_atari.py

tests:
	python -m unittest discover -s 'tests' -p '*_tests.py' -v

docs:
	cd docs && pydocmd build && pydocmd serve

publish:
	python setup.py sdist
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
