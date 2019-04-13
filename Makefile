
.PHONY: all tests dist docs

all: sac

# Demo
acp:
	python examples/actor_critic_pendulum.py

reinforce:
	python examples/reinforce_cartpole.py

ac:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python examples/actor_critic_cartpole.py

grid:
	python examples/actor_critic_gridworld.py

# Atari
dist-a2c:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python -m torch.distributed.launch \
	          --nproc_per_node=16 \
		    examples/atari/dist_a2c_atari.py
ppoa:
	OMP_NUM_THREADS=4 \
	MKL_NUM_THREADS=4 \
	python examples/atari/ppo_atari.py

bug:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python examples/atari/debug_atari.py

dqn:
	python examples/atari/dqn_atari.py

# PyBullet
dist-ppo:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python -m torch.distributed.launch \
	          --nproc_per_node=16 \
		    examples/pybullet/dist_ppo_pybullet.py

ppo:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python examples/pybullet/ppo_pybullet.py

sac:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python examples/pybullet/sac_pybullet.py

# Tabular
tabular-q:
	python examples/tabular/sarsa.py

tabular-l:
	python examples/tabular/q_learning.py



# Admin
dev:
	pip install --progress-bar off torch gym >> log_install.txt
	python setup.py develop

tests:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	python -W ignore::DeprecationWarning -m unittest discover -s 'tests' -p '*_tests.py' -v

docs:
	cd docs && pydocmd build && pydocmd serve

docs-deploy:
	cd docs && pydocmd gh-deploy

publish:
	python setup.py sdist
	twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
