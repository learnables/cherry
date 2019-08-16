#!/usr/bin/env python3

"""
**Description**

Helper functions for implementing TRPO.
"""

import torch as th
from torch import autograd
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def hessian_vector_product(loss, parameters, damping=1e-5):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/algorithms/trpo.py)

    **Description**

    Returns a callable that computes the product of the Hessian of loss
    (w.r.t. parameters) with another vector, using Pearlmutter's trick.

    Note that parameters and the argument of the callable can be tensors
    or list of tensors.

    TODO: The reshaping (if arguments are lists) is not efficiently implemented.
          (It requires a copy) A good idea would be to have
          vector_to_shapes(vec, shapes) or similar.

    NOTE: We can not compute the grads with a vector version of the parameters,
          since that vector (created within the function) will be a different
          tree that is was not used in the computation of the loss.
          (i.e. you get 'One of the differentiated tensors was not used.')

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

    Computes x = A^{-1}b using the conjugate gradient algorithm.

    **References**

    1. Nocedal and Wright. 2006. Numerical Optimization, 2nd edition. Springer.
    2. [https://github.com/Kaixhin/spinning-up-basic/blob/master/trpo.py](https://github.com/Kaixhin/spinning-up-basic/blob/master/trpo.py)
    3. [https://github.com/tristandeleu/pytorch-maml-rl](https://github.com/tristandeleu/pytorch-maml-rl)

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
