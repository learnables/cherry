#!/usr/bin/env python3

import cherry as ch
from .base import Wrapper


class Runner(Wrapper):

    """
    Runner wrapper.
    """

    def __init__(self, env):
        super(Runner, self).__init__(env)
        self.env = env
        self._needs_reset = True
        self._current_state = None

    def reset(self, *args, **kwargs):
        self._current_state = self.env.reset(*args, **kwargs)
        self._needs_reset = False
        return self._current_state

    def step(self, action, *args, **kwargs):
        # TODO: Implement it to be compatible with .run()
        raise NotImplementedError('Runner does not currently support step.')

    def run(self,
            get_action,
            steps=None,
            episodes=None,
            render=False):
        """
        Runner wrapper's run method.
        """

        if steps is None:
            steps = float('inf')
        if episodes is None:
            episodes = float('inf')

        replay = ch.ExperienceReplay()
        collected_episodes = 0
        collected_steps = 0
        while True:
            if collected_steps >= steps or collected_episodes >= episodes:
                return replay
            if self._needs_reset:
                self.reset()
            info = {}
            action = tuple(get_action(self._current_state))
            if len(action) == 2:
                info = action[1]
                action = action[0]
            else:
                action = action[0]
            old_state = self._current_state
            state, reward, done, _ = self.env.step(action)
            if done:
                collected_episodes += 1
                self._needs_reset = True
            replay.append(old_state, action, reward, state, done, **info)
            self._current_state = state
            if render:
                self.env.render()
            collected_steps += 1
