#!/usr/bin/env python3

import torch as th
from torch import autograd
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.distributions import kl_divergence

import cherry as ch
from cherry.algorithms import trpo
from cherry.models.robotics import LinearValue

import bsuite
from bsuite import sweep
from bsuite.utils import gym_wrapper
from bsuite.experiments import summary_analysis
from bsuite.logging import csv_load

from tqdm import tqdm
from copy import deepcopy

from policy import Policy

RESULTS_PATH = './examples/bsuite/'
RANDOM_RESULTS_PATH = RESULTS_PATH + 'logs/random/'
TRPO_RESULTS_PATH = RESULTS_PATH + 'logs/trpo/'


def trpo_update(replay, policy, baseline):
    gamma = 0.99
    tau = 0.95
    max_kl = 0.01
    ls_max_steps = 15
    backtrack_factor = 0.5
    old_policy = deepcopy(policy)
    for step in range(10):
        states = replay.state()
        actions = replay.action()
        rewards = replay.reward()
        dones = replay.done()
        next_states = replay.next_state()
        returns = ch.td.discount(gamma, rewards, dones)
        baseline.fit(states, returns)
        values = baseline(states)
        next_values = baseline(next_states)

        # Compute KL
        with th.no_grad():
            old_density = old_policy.density(states)
        new_density = policy.density(states)
        kl = kl_divergence(old_density, new_density).mean()

        # Compute surrogate loss
        old_log_probs = old_density.log_prob(actions).mean(dim=1, keepdim=True)
        new_log_probs = new_density.log_prob(actions).mean(dim=1, keepdim=True)
        bootstraps = values * (1.0 - dones) + next_values * dones
        advantages = ch.pg.generalized_advantage(gamma, tau, rewards,
                                                 dones, bootstraps, th.zeros(1))
        advantages = ch.normalize(advantages).detach()
        surr_loss = trpo.policy_loss(new_log_probs, old_log_probs, advantages)

        # Compute the update
        grad = autograd.grad(surr_loss,
                             policy.parameters(),
                             retain_graph=True)
        Fvp = trpo.hessian_vector_product(kl, policy.parameters())
        grad = parameters_to_vector(grad).detach()
        step = trpo.conjugate_gradient(Fvp, grad)
        lagrange_mult = 0.5 * th.dot(step, Fvp(step)) / max_kl
        step = step / lagrange_mult
        step_ = [th.zeros_like(p.data) for p in policy.parameters()]
        vector_to_parameters(step, step_)
        step = step_

        #  Line-search
        for ls_step in range(ls_max_steps):
            stepsize = backtrack_factor**ls_step
            clone = deepcopy(policy)
            for c, u in zip(clone.parameters(), step):
                c.data.add_(-stepsize, u.data)
            new_density = clone.density(states)
            new_kl = kl_divergence(old_density, new_density).mean()
            new_log_probs = new_density.log_prob(actions).mean(dim=1, keepdim=True)
            new_loss = trpo.policy_loss(new_log_probs, old_log_probs, advantages)
            if new_loss < surr_loss and new_kl < max_kl:
                for p, c in zip(policy.parameters(), clone.parameters()):
                    p.data[:] = c.data[:]
                break


def run_trpo():
    ch.debug.debug()
    for i, env_name in enumerate(sweep.SWEEP):
        dm_env = bsuite.load_and_record_to_csv(env_name,
                                               results_dir=TRPO_RESULTS_PATH,
                                               overwrite=True)

        #  Instanciate the env and agent
        env = gym_wrapper.GymWrapper(dm_env)
        env = ch.envs.Torch(env)
        env = ch.envs.Runner(env)
        policy = Policy(env)
        baseline = LinearValue(env.state_size)

        #  Generate the results
        replay = ch.ExperienceReplay()
        for episode in tqdm(range(1, 1+ env.bsuite_num_episodes), desc=env_name):
            replay += env.run(policy, episodes=1)
            if episode % 10 == 0:
                trpo_update(replay, policy, baseline)
                replay.empty()


def run_random():
    for env_name in sweep.SWEEP:  #  Or for a specific suite: sweep.DEEP_SEA
        dm_env = bsuite.load_and_record_to_csv(env_name,
                                               results_dir=RANDOM_RESULTS_PATH,
                                               overwrite=True)

        #  Instanciate the agent
        env = gym_wrapper.GymWrapper(dm_env)
        env = ch.envs.Runner(env)
        policy = ch.models.RandomPolicy(env)

        #  Generate the results
        print('Running', env_name)
        env.run(policy, episodes=env.bsuite_num_episodes)


def make_plots():
    # Setup plotting
    import pandas as pd
    import plotnine as gg
    import warnings
    import matplotlib.pyplot as plt

    pd.options.mode.chained_assignment = None
    gg.theme_set(gg.theme_bw(base_size=16, base_family='serif'))
    gg.theme_update(figure_size=(12, 8), panel_spacing_x=0.5, panel_spacing_y=0.5)
    warnings.filterwarnings('ignore')

    #  Load Results
    experiments = {
        'Random': RANDOM_RESULTS_PATH,
        'TRPO': TRPO_RESULTS_PATH,
    }
    data_frame, sweep_vars = csv_load.load_bsuite(experiments)
    bsuite_score = summary_analysis.bsuite_score(data_frame,
                                                 sweep_vars)
    bsuite_summary = summary_analysis.ave_score_by_tag(bsuite_score,
                                                       sweep_vars)

    #  Generate the plots
    radar_fig = summary_analysis.bsuite_radar_plot(bsuite_summary, sweep_vars)
    bar_fig = summary_analysis.bsuite_bar_plot(bsuite_score, sweep_vars)
    compare_bar_fig = summary_analysis.bsuite_bar_plot_compare(bsuite_score,
                                                               sweep_vars)

    radar_fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=-10.0)
    radar_fig.savefig(RESULTS_PATH + 'radar_fig.png', bbox_inches='tight')
    bar_fig.save(RESULTS_PATH + 'bar_fig.png')
    compare_bar_fig.save(RESULTS_PATH + 'compare_bar_fig.png')


def main():
#    run_random()
    run_trpo()
#    make_plots()


if __name__ == '__main__':
    main()
