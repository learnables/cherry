# DeepMind Control

![](./results/all.png)

## Installs


```
pip install gym==0.25.0 metaschool simple-parsing wandb dm-control
pip install git+https://github.com/denisyarats/dmc2gym.git
pip install plotify  # optional: for plots only
```

## Run

SAC (proprioceptive states):
```
make GPU=0 DOMAIN=cartpole TASK=swingup sac
```

DrQ (vision states):
```
make GPU=0 DOMAIN=cartpole TASK=swingup drq
```

DrQv2 (vision states):
```
make GPU=0 DOMAIN=cartpole TASK=swingup drqv2
```

## Sweeps

The results in the figure above are obtained by running the following sweeps.

For SAC: `dmc_sweep_proprioceptive.yaml`

For DrQ/DrQv2: `dmc_sweep_vision.yaml`

**Note** We set `log_alpha=-10` and `use_automatic_entropy_tuning=0` only for SAC/DrQ on Cheetah and Walker.
