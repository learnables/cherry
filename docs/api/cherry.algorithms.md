
# cherry.algorithms

::: cherry.algorithms.AlgorithmArguments
    selection:
      members:
        - __init__

::: cherry.algorithms.A2C
    selection:
      members:
        - state_value_loss
        - policy_loss

::: cherry.algorithms.DDPG
    selection:
      members:
        - state_value_loss

::: cherry.algorithms.DrQ
    selection:
      members:
        - __init__
        - update
        - action_value_loss
        - policy_loss

::: cherry.algorithms.DrQv2
    selection:
      members:
        - __init__
        - update
        - action_value_loss
        - policy_loss

::: cherry.algorithms.PPO
    selection:
      members:
        - __init__
        - update
        - state_value_loss
        - policy_loss

::: cherry.algorithms.TD3
    selection:
      members:
        - __init__
        - update

::: cherry.algorithms.TRPO
    selection:
      members:
        - policy_loss
        - hessian_vector_product
        - conjugate_gradient
        - line_search

::: cherry.algorithms.SAC
    selection:
      members:
        - __init__
        - update
        - action_value_loss
        - policy_loss
