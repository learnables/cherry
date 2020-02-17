import unittest
import gym
import cherry

class VecEnvTests(unittest.TestCase):
    def test_vecenv(self):
        gym_env = gym.vector.make('CartPole-v0', num_envs=2)
        cherry_env = cherry.envs.Torch(gym_env)
        
        assert cherry_env.action_size == gym_env.action_space[0].n
        assert cherry_env.state_size == gym_env.observation_space.shape[1]
        
        assert gym_env.reset().shape == cherry_env.reset().shape


        gym_env = gym.vector.make('CartPole-v0', num_envs=3)
        cherry_env = cherry.envs.Torch(gym_env)
        
        assert cherry_env.action_size == gym_env.action_space[0].n
        assert cherry_env.state_size == gym_env.observation_space.shape[1]
        
        assert gym_env.reset().shape == cherry_env.reset().shape

if __name__ == '__main__':
    unittest.main()