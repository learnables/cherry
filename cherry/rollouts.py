#!/usr/bin/env python3

from itertools import count


def collect(env,
            get_action,
            replay,
            num_steps=None,
            num_episodes=None,
            render=False):
    """
    This functions collects experience and stores it into a
    ExperienceReplay.
    """

    if num_steps is None:
        num_steps = float('inf')
    if num_episodes is None:
        num_episodes = float('inf')

    collected_samples = 0
    for episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            info = {}
            action = tuple(get_action(state)) 
            if len(action) == 2:
                info = action[1]
                action = action[0]
            else:
                action = action[0]
            old_state = state
            state, reward, done, _ = env.step(action)
            replay.add(old_state, action, reward, state, done, info=info)
            if render:
                env.render()
            if done:
                break
            collected_samples += 1
            if collected_samples >= num_steps:
                return collected_samples, collected_samples
        if episode >= num_episodes:
                return collected_samples, collected_samples
