GPU=0
GL=egl # or osmesa
DOMAIN=cartpole
TASK=swingup

all: drq

drq:
	OMP_NUM_THREADS=6 \
	MKL_NUM_THREADS=6 \
	CUDA_VISIBLE_DEVICES=$(GPU) \
	MUJOCO_GL=$(GL) \
	python train.py \
		--options.algorithm='drq' \
		--options.num_updates=200000 \
		--options.replay_size=1000000 \
		--options.deterministic_eval \
		--tasks.domain_name=$(DOMAIN) \
		--tasks.task_name=$(TASK) \
		--tasks.vision_states=1 \
		--policy.projector_size=50 \
		--qvalue.projector_size=50 \
		--drq.target_polyak_weight=0.995 \
		--drq.batch_size=512 \
		--options.log_wandb \
		--options.seed=1234

sac:
	OMP_NUM_THREADS=6 \
	MKL_NUM_THREADS=6 \
	CUDA_VISIBLE_DEVICES=$(GPU) \
	MUJOCO_GL=$(GL) \
	python train.py \
		--options.algorithm='sac' \
		--options.num_updates=200000 \
		--options.replay_size=1000000 \
		--options.deterministic_eval \
		--options.log_alpha=-1.0 \
		--tasks.domain_name=$(DOMAIN) \
		--tasks.task_name=$(TASK) \
		--tasks.vision_states=0 \
		--policy.projector_size=50 \
		--qvalue.projector_size=50 \
		--sac.target_polyak_weight=0.995 \
		--sac.batch_size=512 \
		--sac.use_automatic_entropy_tuning=False \
		--options.log_wandb \
		--options.seed=1234

drqv2:
	OMP_NUM_THREADS=6 \
	MKL_NUM_THREADS=6 \
	CUDA_VISIBLE_DEVICES=$(GPU) \
	MUJOCO_GL=$(GL) \
	python train.py \
		--options.algorithm='drqv2' \
		--options.num_updates=200000 \
		--options.replay_size=1000000 \
		--options.deterministic_eval \
		--tasks.domain_name=$(DOMAIN) \
		--tasks.task_name=$(TASK) \
		--tasks.vision_states=1 \
		--policy.projector_size=50 \
		--qvalue.projector_size=50 \
		--drqv2.target_polyak_weight=0.995 \
		--drqv2.batch_size=512 \
		--drqv2.nsteps=1 \
		--drqv2.std_decay=0.999954 \
		--drqv2.min_std=0.1 \
		--options.log_wandb \
		--options.seed=1234

plots:
	python ./results/dmc_plot_results.py
