# -*- coding=utf-8 -*-

import torch


class Policy(torch.nn.Module):

    """
    [[Source]](https://github.com/learnables/cherry/blob/master/cherry/nn/policy.py)

    Boilerplate Module to represent policies.

    **Example**

    ~~~python
    class RandomPolicy(Policy):

        def __init__(self, num_actions=5):
            self.num_actions = num_actions

        def forward(self, state):  # must return a density
            probs = torch.ones(self.num_actions) / self.num_actions
            density = cherry.distributions.Categorical(probs=probs)
            return density

    # We can now use some predefined functions:
    random_policy = RandomPolicy()
    actions = random_policy.act(states, deterministic=True)
    log_probs = random_policy.log_probs(states, actions)
    ~~~

    """

    def __init__(self):
        super(Policy, self).__init__()

    def log_prob(self, state, action):
        density = self(state)
        return density.log_prob(action)

    def act(self, state, deterministic=False):
        density = self(state)
        if deterministic:
            return density.mode()
        else:
            return density.sample()


if __name__ == '__main__':
    class CatPolicy(Policy):

        def forward(self, state):
            logits = torch.randn_like(state)
            return torch.distributions.Categorical(logits=logits)

    state = torch.randn(5, 3)
    cat_policy = CatPolicy()
    cat_density = cat_policy(state)
    cat_action = cat_policy.act(state, deterministic=True)
    cat_logprob = cat_policy.log_prob(state, cat_action)

    class NormalPolicy(Policy):

        def forward(self, state):
            loc = torch.randn(state.shape[-1])
            scale = torch.randn(state.shape[-1])
            return torch.distributions.Normal(loc=loc, scale=scale)

    state = torch.randn(5, 3)
    policy = NormalPolicy()
    density = policy(state)
    action = policy.act(state, deterministic=True)
    logprob = policy.log_prob(state, action)
    print(logprob.shape)
