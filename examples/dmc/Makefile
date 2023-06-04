GPU=0
GL=egl # or osmesa

all: drq

drq:
	OMP_NUM_THREADS=6 \
	MKL_NUM_THREADS=6 \
	CUDA_VISIBLE_DEVICES=$(GPU) \
	MUJOCO_GL=$(GL) \
	python train.py \
		--options.algorithm='drq' \
		--tasks.vision_states=1 \
		--drq.target_polyak_weight=0.995 \
		--drq.batch_size=512 \
		--policy.projector_size=50 \
		--qvalue.projector_size=50 \
		--options.seed=1234

sac:
	python train.py

drqv2:
	python train.py
