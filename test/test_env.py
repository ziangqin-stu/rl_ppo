import gym
from gym_minigrid.wrappers import *
import matplotlib.pyplot as plt

# env = gym.make('MiniGrid-Empty-5x5-v0')
# env = gym.make('CartPole-v1')
env = gym.make('Humanoid-v2')
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


