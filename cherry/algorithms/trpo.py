#!/usr/bin/env python3

"""
**Description**

Helper functions for implementing Trust-Region Policy Optimization.

Recall that TRPO strives to solve the following objective:

$$
\\max_\\theta \\quad \\mathbb{E}\\left[ \\frac{\\pi_\\theta}{\\pi_\\text{old}} \\cdot A  \\right] \\\\
\\text{subject to} \\quad D_\\text{KL}(\\pi_\\text{old} \\vert \\vert \\pi_\\theta) \\leq \\delta.
$$


"""

import torch as th
from torch import autograd
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from cherry import debug


def policy_loss(new_log_probs, old_log_probs, advantages):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/a2c.py)

    **Description**

    The policy loss for Trust-Region Policy Optimization.

    This is also known as the surrogate loss.

    **References**

    1. Schulman et al. 2015. “Trust Region Policy Optimization.” ICML 2015.

    **Arguments**

    * **new_log_probs** (tensor) - The log-density of actions from the target policy.
    * **old_log_probs** (tensor) - The log-density of actions from the behaviour policy.
    * **advantages** (tensor) - Advantage of the actions.

    **Returns**

    * (tensor) - The policy loss for the given arguments.

    **Example**

    ~~~python
    advantage = ch.pg.generalized_advantage(GAMMA,
                                            TAU,
                                            replay.reward(),
                                            replay.done(),
                                            replay.value(),
                                            next_state_value)
    new_densities = policy(replay.state())
    new_logprobs = new_densities.log_prob(replay.action())
    loss = policy_loss(new_logprobs,
                       replay.logprob().detach(),
                       advantage.detach())
    ~~~
    """
    msg = 'log_probs and advantages must have equal size.'
    assert new_log_probs.size() == old_log_probs.size() == advantages.size(), msg
    if debug.IS_DEBUGGING:
        if old_log_probs.requires_grad:
            debug.logger.warning('TRPO:policy_loss: old_log_probs.requires_grad is True.')
        if advantages.requires_grad:
            debug.logger.warning('TRPO:policy_loss: advantages.requires_grad is True.')
        if not new_log_probs.requires_grad:
            debug.logger.warning('TRPO:policy_loss: new_log_probs.requires_grad is False.')
    ratio = th.exp(new_log_probs - old_log_probs)
    return - th.mean(ratio * advantages)


def hessian_vector_product(loss, parameters, damping=1e-5):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/trpo.py)

    **Description**

    Returns a callable that computes the product of the Hessian of loss
    (w.r.t. parameters) with another vector, using Pearlmutter's trick.

    Note that parameters and the argument of the callable can be tensors
    or list of tensors.

    **References**

    1. Pearlmutter, B. A. 1994. “Fast Exact Multiplication by the Hessian.” Neural Computation.

    **Arguments**

    * **loss** (tensor) - The loss of which to compute the Hessian.
    * **parameters** (tensor or list) - The tensors to take the gradient with respect to.
    * **damping** (float, *optional*, default=1e-5) - Damping of the Hessian-vector product.

    **Returns**

    * **hvp(other)** (callable) - A function to compute the Hessian-vector product,
        given a vector or list `other`.

    **Example**

    ~~~python
    pass
    ~~~
    """
    if not isinstance(parameters, th.Tensor):
        parameters = list(parameters)
    grad_loss = autograd.grad(loss,
                              parameters,
                              create_graph=True,
                              retain_graph=True)
    grad_loss = parameters_to_vector(grad_loss)

    def hvp(other):
        """
        TODO: The reshaping (if arguments are lists) is not efficiently implemented.
              (It requires a copy) A good idea would be to have
              vector_to_shapes(vec, shapes) or similar.

        NOTE: We can not compute the grads with a vector version of the parameters,
              since that vector (created within the function) will be a different
              tree that is was not used in the computation of the loss.
              (i.e. you get 'One of the differentiated tensors was not used.')
        """
        shape = None
        if not isinstance(other, th.Tensor):
            shape = [th.zeros_like(o) for o in other]
            other = parameters_to_vector(other)
        grad_prod = th.dot(grad_loss, other)
        hessian_prod = autograd.grad(grad_prod,
                                     parameters,
                                     retain_graph=True)
        hessian_prod = parameters_to_vector(hessian_prod)
        hessian_prod = hessian_prod + damping * other
        if shape is not None:
            vector_to_parameters(hessian_prod, shape)
            hessian_prod = shape
        return hessian_prod

    return hvp


def conjugate_gradient(Ax, b, num_iterations=10, tol=1e-10, eps=1e-8):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/trpo.py)

    **Description**

    Computes \\(x = A^{-1}b\\) using the conjugate gradient algorithm.

    **Credit**

    Adapted from Kai Arulkumaran's implementation, with additions inspired from John Schulman's implementation.

    **References**

    1. Nocedal and Wright. 2006. "Numerical Optimization, 2nd edition". Springer.
    2. Shewchuk et al. 1994. “An Introduction to the Conjugate Gradient Method without the Agonizing Pain.” CMU.

    **Arguments**

    * **Ax** (callable) - Given a vector x, computes A@x.
    * **b** (tensor or list) - The reference vector.
    * **num_iterations** (int, *optional*, default=10) - Number of conjugate gradient iterations.
    * **tol** (float, *optional*, default=1e-10) - Tolerance for proposed solution.
    * **eps** (float, *optional*, default=1e-8) - Numerical stability constant.

    **Returns**

    * **x** (tensor or list) - The solution to Ax = b, as a list if b is a list else a tensor.

    **Example**

    ~~~python
    pass
    ~~~
    """
    shape = None
    if not isinstance(b, th.Tensor):
        shape = [th.zeros_like(b_i) for b_i in b]
        b = parameters_to_vector(b)
    x = th.zeros_like(b)
    r = b
    p = r
    r_dot_old = th.dot(r, r)
    for _ in range(num_iterations):
        Ap = Ax(p)
        alpha = r_dot_old / (th.dot(p, Ap) + eps)
        x += alpha * p
        r -= alpha * Ap
        r_dot_new = th.dot(r, r)
        p = r + (r_dot_new / r_dot_old) * p
        r_dot_old = r_dot_new
        if r_dot_new.item() < tol:
            break
    if shape is not None:
        vector_to_parameters(x, shape)
        x = shape
    return x
