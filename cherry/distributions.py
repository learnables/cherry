#!/usr/bin/env python3

"""
**Description**

A set of common distributions.

"""

import torch as th
import torch.nn as nn
import cherry as ch

from torch.distributions import Distribution


class Categorical(th.distributions.Categorical):

    """
    <a href="https://github.com/seba-1511/cherry/blob/master/cherry/distributions.py" class="source-link">[Source]</a>

    ## Description

    Similar to `torch.nn.Categorical`, but reshapes tensors of N
    samples into (N, 1)-shaped tensors.

    ## Arguments

    Identical to `torch.distribution.Categorical`.

    ## Example

    ~~~python
    dist = Categorical(logits=torch.randn(bsz, action_size))
    actions = dist.sample()  # shape: bsz x 1
    log_probs = dist.log_prob(actions)  # shape: bsz x 1
    deterministic_action = action.mode()
    ~~~
    """

    def sample(self):
        samples = super(Categorical, self).sample()
        return samples.unsqueeze(-1)

    def log_prob(self, x):
        log_prob = super(Categorical, self).log_prob(x.reshape(-1))
        return log_prob.reshape_as(x)

    def mode(self):
        """
        ## Description

        Returns the model of normal distribution (ie, argmax over probabilities).
        """
        return self.probs.argmax(dim=-1, keepdim=True)


class Normal(th.distributions.Normal):

    """
    <a href="https://github.com/seba-1511/cherry/blob/master/cherry/distributions.py" class="source-link">[Source]</a>

    ## Description

    Similar to PyTorch's `Independent(Normal(loc, std))`: when computing log-densities or the entropy,
    we sum over the last dimension.

    This is typically used to compute log-probabilities of N-dimensional actions
    sampled from a multivariate Gaussian with diagional covariance.

    ## Arguments

    Identical to `torch.distribution.Normal`.

    ## Example

    ~~~python
    normal = Normal(torch.zeros(bsz, action_size), torch.ones(bsz, action_size))
    actions = normal.sample()
    log_probs = normal.log_prob(actions)  # shape: bsz x 1
    entropies = normal.entropy()  # shape: bsz x 1
    deterministic_action = action.mode()
    ~~~
    """

    def log_prob(self, x):
        log_prob = super(Normal, self).log_prob(x)
        return log_prob.sum(-1, keepdim=True)

    def entropy(self):
        entropy = super(Normal, self).entropy().sum(dim=-1, keepdim=True)
        return entropy

    def mode(self):
        """
        ## Description

        Returns the model of normal distribution (ie, its mean).
        """
        return self.mean


class Reparameterization(object):

    """
    <a href="https://github.com/seba-1511/cherry/blob/master/cherry/distributions.py" class="source-link">[Source]</a>

    ## Description

    Unifies interface for distributions that support `rsample` and those that do not.

    When calling `sample()`, this class checks whether `density` has a `rsample()` member,
    and defaults to call `sample()` if it does not.

    ## References

    1. Kingma and Welling. 2013. “Auto-Encoding Variational Bayes.” arXiv [stat.ML].

    ## Example

    ~~~python
    density = Normal(mean, std)
    reparam = Reparameterization(density)
    sample = reparam.sample()  # Uses Normal.rsample()
    ~~~

    """

    def __init__(self, density):
        """
        ## Arguments

        * `density` (Distribution) - The distribution to wrap.
        """
        self.density = density

    def sample(self, *args, **kwargs):
        if self.density.has_rsample:
            return self.density.rsample(*args, **kwargs)
        return self.density.sample(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.density, name)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'Reparameterization(' + str(self.density) + ')'


class ActionDistribution(nn.Module):
    """
    <a href="https://github.com/seba-1511/cherry/blob/master/cherry/distributions.py" class="source-link">[Source]</a>

    ## Description

    A helper module to automatically choose the proper policy distribution,
    based on the Gym environment `action_space`.

    For `Discrete` action spaces, it uses a `Categorical` distribution, otherwise
    it uses a `Normal` which uses a diagonal covariance matrix.

    This class enables to write single version policy body that will be compatible
    with a variety of environments.

    ## Example

    ~~~python
    env = gym.make('CartPole-v1')
    action_dist = ActionDistribution(env)
    ~~~

    """

    def __init__(self, env, logstd=None, use_probs=False, reparam=False):
        """
        ## Arguments

        * `env` (Environment) - Gym environment for which actions will be sampled.
        * `logstd` (float/tensor, *optional*, default=0) - The log standard deviation for the `Normal` distribution.
        * `use_probs` (bool, *optional*, default=False) - Whether to use probabilities or logits for the `Categorical` case.
        * `reparam` (bool, *optional*, default=False) - Whether to use reparameterization in the `Normal` case.
        """
        super(ActionDistribution, self).__init__()
        self.use_probs = use_probs
        self.reparam = reparam
        self.is_discrete = ch.envs.is_discrete(env.action_space)
        if not self.is_discrete:
            if logstd is None:
                action_size = ch.envs.get_space_dimension(env.action_space,
                                                          vectorized_dims=False)
                logstd = nn.Parameter(th.zeros(action_size))
            if isinstance(logstd, (float, int)):
                logstd = nn.Parameter(th.Tensor([logstd]))
            self.logstd = logstd

    def forward(self, x):
        if self.is_discrete:
            if self.use_probs:
                return Categorical(probs=x)
            return Categorical(logits=x)
        else:
            density = Normal(loc=x, scale=self.logstd.exp())
            if self.reparam:
                density = Reparameterization(density)
            return density


class TanhNormal(Distribution):

    """
    <a href="https://github.com/seba-1511/cherry/blob/master/cherry/distributions.py" class="source-link">[Source]</a>

    ## Description

    Implements a Normal distribution followed by a Tanh, often used with the Soft Actor-Critic.

    This implementation also exposes `sample_and_log_prob` and `rsample_and_log_prob`,
    which returns both samples and log-densities.
    The log-densities are computed using the pre-activation values for numerical stability.

    ## References

    1. Haarnoja et al. 2018. “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.” arXiv [cs.LG].
    2. Haarnoja et al. 2018. “Soft Actor-Critic Algorithms and Applications.” arXiv [cs.LG].
    3. Vitchyr Pong's [RLkit](https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/distributions.py).

    **Example**
    ~~~python
    mean = th.zeros(5)
    std = th.ones(5)
    dist = TanhNormal(mean, std)
    samples = dist.rsample()
    logprobs = dist.log_prob(samples)  # Numerically unstable :(
    samples, logprobs = dist.rsample_and_log_prob()  # Stable :)
    ~~~

    """

    def __init__(self, normal_mean, normal_std):
        """
        ## Arguments

        * `normal_mean` (tensor) - Mean of the Normal distribution.
        * `normal_std` (tensor) - Standard deviation of the Normal distribution.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)

    def sample_n(self, n):
        z = self.normal.sample_n(n)
        return th.tanh(z)

    def log_prob(self, value):
        pre_tanh_value = (th.log1p(value) - th.log1p(-value)).mul(0.5)
        offset = th.log1p(-value**2 + 1e-6)
        return self.normal.log_prob(pre_tanh_value) - offset

    def sample(self):
        z = self.normal.sample().detach()
        return th.tanh(z)

    def mean(self):
        """
        ## Description

        Returns the mean of the TanhDistribution (ie, tan(normal.mean)).
        """
        return th.tanh(self.normal.mean)

    def mode(self):
        """
        ## Description

        Returns the mode of the TanhDistribution (ie, its mean).
        """
        return self.mean()

    def sample_and_log_prob(self):
        """
        ## Description

        Samples from the TanhNormal and computes the log-density of the
        samples in a numerically stable way.

        ## Returns

        * `value` (tensor) - samples from the TanhNormal.
        * `log_prob` (tensor) - log-probabilities of the samples.

        ## Example

        ~~~python
        tanh_normal = TanhNormal(torch.zeros(bsz, action_size), torch.ones(bsz, action_size))
        actions, log_probs = tanh_normal.sample_and_log_prob()
        ~~~
        """
        z = self.normal.sample().detach()
        value = th.tanh(z)
        offset = th.log1p(-value**2 + 1e-6)
        log_prob = self.normal.log_prob(z) - offset
        return value, log_prob

    def rsample_and_log_prob(self):
        """
        ## Description

        Similar to `sample_and_log_prob` but with reparameterized samples.
        """
        z = self.normal.rsample()
        value = th.tanh(z)
        offset = th.log1p(-value**2 + 1e-6)
        log_prob = self.normal.log_prob(z) - offset
        return value, log_prob

    def rsample(self):
        z = self.normal.rsample()
        z.requires_grad_()
        return th.tanh(z)
