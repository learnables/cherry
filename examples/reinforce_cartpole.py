#!/usr/bin/env python3

import gym

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())
