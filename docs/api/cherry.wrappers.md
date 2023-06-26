
# cherry.wrappers

::: cherry.wrappers.Wrapper
    selection:
      members:
        - is_vectorized
        - discrete_action
        - discrete_state
        - action_size
        - state_size

::: cherry.wrappers.Runner
    selection:
      members:
        - run

::: cherry.wrappers.Torch
    selection:
      members:
        - __init__

::: cherry.wrappers.RewardClipper
    selection:
      members:
        - __init__

::: cherry.wrappers.AddTimestep
    selection:
      members:
        - __init__

::: cherry.wrappers.ActionSpaceScaler
    selection:
      members:
        - __init__

## Soon Deprecated

!!! info
    The following wrappers will soon be deprecated because they are available in `gym`.

::: cherry.wrappers.Logger
    selection:
      members:
        - __init__

