#!/usr/bin/env bash

# Q-Learning
python benchmark.py ../examples/tabular/q_learning.py CliffWalking-v0 42
python benchmark.py ../examples/tabular/q_learning.py FrozenLake-v0 42
python benchmark.py ../examples/tabular/q_learning.py BlackJack-v0 42
python benchmark.py ../examples/tabular/q_learning.py NChain-v0 42

# SARSA
python benchmark.py ../examples/tabular/sarsa.py CliffWalking-v0 42
python benchmark.py ../examples/tabular/sarsa.py FrozenLake-v0 42
python benchmark.py ../examples/tabular/sarsa.py BlackJack-v0 42
python benchmark.py ../examples/tabular/sarsa.py NChain-v0 42
