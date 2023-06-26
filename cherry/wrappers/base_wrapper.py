#!/usr/bin/env python3

import gym

from cherry.envs.utils import get_space_dimension, is_vectorized, is_discrete


class Wrapper(gym.Wrapper):

    """
    <a href="" class="source-link">[Source]</a>

    ## Description

    This class allows to chain Environment Wrappers while still being able to
    access the properties of wrapped wrappers.

    ## Example

    ~~~python
    env = gym.make('MyEnv-v0')
    env = cherry.wrappers.Logger(env)
    env = cherry.wrappers.Runner(env)
    env.log('asdf', 23)  # Uses log() method from cherry.wrappers.Logger.
    ~~~

    """

    @property
    def is_vectorized(self):
        """
        ## Description

        Returns whether the env is vectorized or not.
        """
        return is_vectorized(self)

    @property
    def discrete_action(self):
        """
        ## Description

        Returns whether the env is vectorized or not.
        """
        return is_discrete(self.action_space)

    @property
    def discrete_state(self):
        """
        ## Description

        Returns whether the env is vectorized or not.
        """
        return is_discrete(self.observation_space)

    @property
    def state_size(self):
        """
        ## Description

        The (flattened) size of a single state.
        """
        # Since we want the underlying dimension, vec_dim=False
        return get_space_dimension(self.observation_space,
                                   vectorized_dims=False)

    @property
    def action_size(self):
        """
        ## Description

        The number of dimensions of a single action.
        """
        # Since we want the underlying dimension, vec_dim=False
        return get_space_dimension(self.action_space,
                                   vectorized_dims=False)

    def __getattr__(self, attr):
        if attr in self.__dict__.keys():
            return getattr(self, attr)
        else:
            return getattr(self.env, attr)
