
.PHONY: all tests dist docs

all: sac

dist:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python -m torch.distributed.launch \
	          --nproc_per_node=16 \
		    examples/dist_a2c_atari.py
bug:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python examples/debug_atari.py

tabular:
	python examples/tabular/sarsa.py
#	python examples/tabular/q_learning.py

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

dev:
	pip install --progress-bar off torch gym >> log_install.txt
	python setup.py develop

tests:
	python -W ignore::DeprecationWarning -m unittest discover -s 'tests' -p '*_tests.py' -v

docs:
	cd docs && pydocmd build && pydocmd serve

docs-deploy:
	cd docs && pydocmd gh-deploy

publish:
	python setup.py sdist
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
