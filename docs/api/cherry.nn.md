
# cherry.nn

::: cherry.nn.Policy
    selection:
      members:
        - __init__
        - forward
        - log_prob
        - act

::: cherry.nn.ActionValue
    selection:
      members:
        - forward
        - all_action_values

::: cherry.nn.Twin
    selection:
      members:
        - __init__
        - forward
        - twin

::: cherry.nn.RoboticsLinear
    selection:
      members:
        - __init__

::: cherry.nn.EpsilonGreedy
    selection:
      members:
        - __init__

::: cherry.nn.MLP
    selection:
      members:
        - __init__

::: cherry.nn.Lambda
    selection:
      members:
        - __init__
