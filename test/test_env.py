import gym
from gym_minigrid.wrappers import *
# import matplotlib.pyplot as plt
import numpy as np
import torch


# env = gym.make('MiniGrid-Empty-5x5-v0')
# env = gym.make('CartPole-v1')
def test_minigrid_wrapper(env_name):
    env = gym.make(env_name)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    obs = env.reset()
    done = False
    step = 10
    while not done and step > 0:
        obs, reward, done, info = env.step(env.action_space.sample())
        plt.imshow(obs)
        plt.show()
        step -= 1
    plt.imshow(obs)
    plt.show()


def test_env_nan(env_name, step):
    env = gym.make(env_name)
    done = False
    obs = env.reset()
    while not done and step > 0:
        # action = env.action_space.sample()
        action = (np.random.rand(3) - 0.5) * 2
        assert (env.action_space.low < np.array(torch.Tensor(action))).all() and (
                    np.array(torch.Tensor(action)) < env.action_space.high).all()
        print(action)
        obs, reward, done, info = env.step(action)
        step -= 1


test_env_nan('CartPole-v1', 100)
