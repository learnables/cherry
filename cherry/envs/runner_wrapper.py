#!/usr/bin/env python3

import cherry as ch
from .base import Wrapper

from collections.abc import Iterable


class Runner(Wrapper):

    """
    Runner wrapper.
    """

    def __init__(self, env):
        super(Runner, self).__init__(env)
        self.env = env
        self._needs_reset = True
        self._current_state = None
        self.is_vectorized = hasattr(env, 'num_envs')

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
            if self.is_vectorized:
                raise Exception('Can not use episodes with vectorized environments.')
        elif episodes is None:
            episodes = float('inf')
        else:
            msg = 'Either steps or episodes should be set.'
            raise Exception(msg)

        replay = ch.ExperienceReplay()
        collected_episodes = 0
        collected_steps = 0
        while True:
            if collected_steps >= steps or collected_episodes >= episodes:
                return replay
            if self._needs_reset:
                self.reset()
            info = {}
            action = get_action(self._current_state)
            if isinstance(action, tuple):
                skip_unpack = False
                if self.is_vectorized:
                    if len(action) > 2:
                        skip_unpack = True
                    elif len(action) == 2 and \
                            self.env.num_envs == 2 and \
                            not isinstance(action[1], dict):
                                # action[1] is not info but an action
                                action = (action, )

                if not skip_unpack:
                    if len(action) == 2:
                        info = action[1]
                        action = action[0]
                    elif len(action) == 1:
                        action = action[0]
                    else:
                        msg = 'get_action should return 1 or 2 values.'
                        raise NotImplementedError(msg)
            old_state = self._current_state
            state, reward, done, _ = self.env.step(action)
            if not isinstance(done, Iterable) and done:
                collected_episodes += 1
                self._needs_reset = True
            replay.append(old_state, action, reward, state, done, **info)
            self._current_state = state
            if render:
                self.env.render()
            collected_steps += 1
