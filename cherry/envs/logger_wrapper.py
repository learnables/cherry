#!/usr/bin/env python3

from gym import Wrapper

from statistics import mean, pstdev


class Logger(Wrapper):

    """
    Tracks and prints some common statistics about the environment.

    TODO:
        * Make it compatible with the (future) Parallel wrapper.
    """

    def __init__(self, env, interval=1000):
        super(Logger, self).__init__(env)
        self.num_steps = 0
        self.num_episodes = 0
        self.all_rewards = []
        self.all_dones = []
        self.interval = interval

    def stats(self):
        rewards = self.all_rewards[-self.interval:]
        episode_rewards = []
        episode_lengths = []
        accum = 0.0
        length = 0
        for r, d in zip(rewards, self.all_dones[-self.interval:]):
            if not d:
                accum += r
                length += 1
            else:
                episode_rewards.append(accum)
                episode_lengths.append(length)
                accum = 0.0
                length = 0
        if length > 0:
                episode_rewards.append(accum)
                episode_lengths.append(length)
        mean_rewards = mean(episode_rewards)
        std_rewards = pstdev(episode_rewards)
        mean_length = int(mean(episode_lengths))
        self.num_episodes += len(episode_rewards)
        msg = '-' * 50 + '\n'
        msg += 'Overall:' + '\n'
        msg += '- Steps: ' + str(self.num_steps) + '\n'
        msg += '- Episodes: ' + str(self.num_episodes) + '\n'
        msg += 'Last ' + str(self.interval) + ' Steps:' + '\n'
        msg += '- Episodes: ' + str(len(episode_rewards)) + '\n'
        msg += '- Mean episode length: ' + str(mean_length) + '\n'
        msg += '- Mean episode reward: ' + '%.2f' % mean_rewards
        msg += ' +/- ' + '%.2f' % std_rewards + '\n'
        msg += '- Mean reward: ' + '%.2f' % mean(rewards)
        msg += ' +/- ' + '%.2f' % pstdev(rewards) + '\n'
        return msg

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        state, reward, done, info = self.env.step(*args, **kwargs)
        self.all_rewards.append(reward)
        self.all_dones.append(done)
        self.num_steps += 1
        if self.interval > 0 and self.num_steps % self.interval == 0:
            print(self.stats())
        return state, reward, done, info
