GPU=0
GL=egl # or osmesa

all: drq

drq:
	OMP_NUM_THREADS=6 \
	MKL_NUM_THREADS=6 \
	CUDA_VISIBLE_DEVICES=$(GPU) \
	MUJOCO_GL=$(GL) \
	python train.py \
		--tasks.vision_states=1 \
		--options.seed=1234

sac:
	python train.py

drqv2:
	python train.py
