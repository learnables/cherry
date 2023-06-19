#!/usr/bin/env python3

import dataclasses
import torch
from torch import autograd
from torch.nn.utils import vector_to_parameters

from cherry import debug
from cherry._utils import _parameters_to_vector
from .arguments import AlgorithmArguments


@dataclasses.dataclass
class TRPO(AlgorithmArguments):

    """
    ## Description

    Helper functions for implementing Trust-Region Policy Optimization.

    Recall that TRPO strives to solve the following objective:

    $$
    \\max_\\theta \\quad \\mathbb{E}\\left[ \\frac{\\pi_\\theta}{\\pi_\\text{old}} \\cdot A  \\right] \\\\
    \\text{subject to} \\quad D_\\text{KL}(\\pi_\\text{old} \\vert \\vert \\pi_\\theta) \\leq \\delta.
    $$

    """

    @staticmethod
    def policy_loss(new_log_probs, old_log_probs, advantages):
        """
        [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/a2c.py)

        ## Description

        The policy loss for Trust-Region Policy Optimization.

        This is also known as the surrogate loss.

        ## References

        1. Schulman et al. 2015. “Trust Region Policy Optimization.” ICML 2015.

        ## Arguments

        * `new_log_probs` (tensor) - The log-density of actions from the target policy.
        * `old_log_probs` (tensor) - The log-density of actions from the behaviour policy.
        * `advantages` (tensor) - Advantage of the actions.

        ## Returns

        * (tensor) - The policy loss for the given arguments.

        ## Example

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
        ratio = torch.exp(new_log_probs - old_log_probs)
        return - torch.mean(ratio * advantages)

    @staticmethod
    def hessian_vector_product(loss, parameters, damping=1e-5):
        """
        [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/trpo.py)

        ## Description

        Returns a callable that computes the product of the Hessian of loss
        (w.r.t. parameters) with another vector, using Pearlmutter's trick.

        Note that parameters and the argument of the callable can be tensors
        or list of tensors.

        ## References

        1. Pearlmutter, B. A. 1994. “Fast Exact Multiplication by the Hessian.” Neural Computation.

        ## Arguments

        * `loss` (tensor) - The loss of which to compute the Hessian.
        * `parameters` (tensor or list) - The tensors to take the gradient with respect to.
        * `damping` (float, *optional*, default=1e-5) - Damping of the Hessian-vector product.

        ## Returns

        * `hvp(other)` (callable) - A function to compute the Hessian-vector product,
            given a vector or list `other`.
        """
        if not isinstance(parameters, torch.Tensor):
            parameters = list(parameters)
        grad_loss = autograd.grad(loss,
                                  parameters,
                                  create_graph=True,
                                  retain_graph=True)
        grad_loss = _parameters_to_vector(grad_loss)

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
            if not isinstance(other, torch.Tensor):
                shape = [torch.zeros_like(o) for o in other]
                other = _parameters_to_vector(other)
            grad_prod = torch.dot(grad_loss, other)
            hessian_prod = autograd.grad(grad_prod,
                                         parameters,
                                         retain_graph=True)
            hessian_prod = _parameters_to_vector(hessian_prod)
            hessian_prod = hessian_prod + damping * other
            if shape is not None:
                vector_to_parameters(hessian_prod, shape)
                hessian_prod = shape
            return hessian_prod

        return hvp

    @staticmethod
    def conjugate_gradient(Ax, b, num_iterations=10, tol=1e-10, eps=1e-8):
        """
        [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/trpo.py)

        ## Description

        Computes \\(x = A^{-1}b\\) using the conjugate gradient algorithm.

        ## Credit

        Adapted from Kai Arulkumaran's implementation, with additions inspired from John Schulman's implementation.

        ## References

        1. Nocedal and Wright. 2006. "Numerical Optimization, 2nd edition". Springer.
        2. Shewchuk et al. 1994. “An Introduction to the Conjugate Gradient Method without the Agonizing Pain.” CMU.

        ## Arguments

        * `Ax` (callable) - Given a vector x, computes A@x.
        * `b` (tensor or list) - The reference vector.
        * `num_iterations` (int, *optional*, default=10) - Number of conjugate gradient iterations.
        * `tol` (float, *optional*, default=1e-10) - Tolerance for proposed solution.
        * `eps` (float, *optional*, default=1e-8) - Numerical stability constant.

        ## Returns

        * `x` (tensor or list) - The solution to Ax = b, as a list if b is a list else a tensor.
        """
        shape = None
        if not isinstance(b, torch.Tensor):
            shape = [torch.zeros_like(b_i) for b_i in b]
            b = _parameters_to_vector(b)
        x = torch.zeros_like(b)
        r = b
        p = r
        r_dot_old = torch.dot(r, r)
        for _ in range(num_iterations):
            Ap = Ax(p)
            alpha = r_dot_old / (torch.dot(p, Ap) + eps)
            x += alpha * p
            r -= alpha * Ap
            r_dot_new = torch.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
            if r_dot_new.item() < tol:
                break
        if shape is not None:
            vector_to_parameters(x, shape)
            x = shape
        return x

    @staticmethod
    def line_search(
        params_init,
        params_update,
        model,
        stop_criterion,
        initial_stepsize=1.0,
        backtrack_factor=0.5,
        max_iterations=15,
    ):
        """
        [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/trpo.py)

        ## Description

        Computes line-search for model parameters given a parameter update and a stopping criterion.

        ## Credit

        Adapted from Kai Arulkumaran's implementation, with additions inspired from John Schulman's implementation.

        ## References

        1. Nocedal and Wright. 2006. "Numerical Optimization, 2nd edition". Springer.

        ## Arguments

        * `params_init` (tensor or iteratble) - Initial parameter values.
        * `params_update` (tensor or iteratble) - Update direction.
        * `model` (Module) - The model to be updated.
        * `stop_criterion` (callable) - Given a model, decided whether to stop the line-search.
        * `initial_stepsize` (float, *optional*, default=1.0) - Initial stepsize of search.
        * `backtrack_factor` (float, *optional*, default=0.5) - Backtracking factor.
        * `max_iterations` (int, *optional*, default=15) - Max number of backtracking iterations.

        ## Returns

        * `new_model` (Module) - The updated model if line-search is successful, else the model with initial parameter values.

        ## Example

        ~~~python
        def ls_criterion(new_policy):
            new_density = new_policy(states)
            new_kl = kl_divergence(old_density, new_densityl).mean()
            new_loss = - qvalue(new_density.sample()).mean()
            return new_loss < policy_loss and new_kl < max_kl

        with torch.no_grad():
            policy = trpo.line_search(
                params_init=policy.parameters(),
                params_update=step,
                model=policy,
                criterion=ls_criterion
            )
        ~~~
        """
        # preprocess inputs
        if not isinstance(params_init, torch.Tensor):
            params_init = _parameters_to_vector(params_init)
        if not isinstance(params_update, torch.Tensor):
            params_update = _parameters_to_vector(params_update)

        # line-search on stepsize
        for iteration in range(max_iterations):
            stepsize = initial_stepsize * backtrack_factor**iteration
            vector_to_parameters(
                params_init - stepsize * params_update,
                model.parameters(),
            )
            if stop_criterion(model):
                return model

        # search failed
        vector_to_parameters(params_init, model.parameters())
        return model


policy_loss = TRPO.policy_loss
hessian_vector_product = TRPO.hessian_vector_product
conjugate_gradient = TRPO.conjugate_gradient
line_search = TRPO.line_search
