import ray
import gym
import time
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from utils import gen_actor, gen_critic, gen_env, get_advantage_new, get_values
from networks import get_norm_log_prob, CriticNet
from rolloutmemory import RolloutMemory

"""
Test RAY Ability
"""


@ray.remote
def single_task_parallel(id):
    print("id={}".format(id))
    time.sleep(1.)
    return (1)


def single_task(task_id):
    print("id={}".format(task_id))
    time.sleep(2.)
    return 1


def parallel_work(work_number):
    time_start = time.time()
    data = [single_task_parallel.remote(i) for i in range(work_number)]
    data = [ray.get(obj_id) for obj_id in data]
    time_end = time.time()
    print("parallel_time: {}, data".format(time_end - time_start, data))


def serial_work(work_number):
    time_start = time.time()
    data = [single_task(i) for i in range(work_number)]
    time_end = time.time()
    print("parallel_time: {}, data".format(time_end - time_start, data))


"""
Further Test on RAY for Proj
"""


class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, action_scale):
        super(Actor, self).__init__()
        # mean
        self.mean_fc1 = nn.Linear(input_size, hidden_dim)
        self.mean_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_fc5 = nn.Linear(hidden_dim, output_size)
        # covariance
        self.cov_fc1 = nn.Linear(input_size, hidden_dim // 2)
        self.cov_fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.cov_fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.cov_fc4 = nn.Linear(hidden_dim // 2, output_size)
        # action scale
        self.scale = torch.tensor([action_scale]).float()
        # initialize network parameters
        nn.init.orthogonal_(self.mean_fc1.weight)
        nn.init.orthogonal_(self.mean_fc2.weight)
        nn.init.orthogonal_(self.mean_fc3.weight)
        nn.init.orthogonal_(self.mean_fc4.weight)
        nn.init.orthogonal_(self.mean_fc5.weight)
        nn.init.orthogonal_(self.cov_fc1.weight)
        nn.init.orthogonal_(self.cov_fc2.weight)
        nn.init.orthogonal_(self.cov_fc3.weight)
        nn.init.orthogonal_(self.cov_fc4.weight)

    def forward(self, state):
        mean = torch.relu(self.mean_fc1(state))
        mean = torch.relu(self.mean_fc2(mean))
        mean = torch.relu(self.mean_fc3(mean))
        mean = torch.relu(self.mean_fc4(mean))
        mean = self.mean_fc5(mean)
        cov = torch.relu(self.cov_fc1(state))
        cov = torch.relu(self.cov_fc2(cov))
        cov = torch.relu(self.cov_fc3(cov))
        cov = torch.exp(self.cov_fc4(cov))
        # if torch.isnan(cov[0]):
        #     print(cov)
        return mean, cov

    def gen_action(self, state):
        mean, cov = self.forward(state)
        dist = Normal(mean, cov)
        raw_action = dist.sample()
        action = self.scale * torch.tanh(raw_action)
        log_prob = get_norm_log_prob([mean, cov], raw_action, self.scale).view(-1)
        return action, log_prob, raw_action

    def policy_out(self, state):
        mean, cov = self.forward(state)
        return mean, cov


# @ray.remote
# class RolloutMemory:
#     """
#     Test:
#         - check data dimension
#         - check data type
#         - check data device
#         - check data value
#     """
#
#     def __init__(self, capacity, env_name):
#         # environment specific parameters
#         self.env = gen_env(env_name)
#         action = self.env.action_space.sample()
#         act_dim = action.shape if isinstance(action, np.ndarray) else (1,)
#         obs_dim = self.env.observation_space.shape
#         # memories
#         self.env_name = env_name
#         self.offset = 0
#         self.capacity = int(capacity)
#         self.old_obs_mem = torch.zeros(capacity, *obs_dim).cuda() if obs_dim[0] > 1 else torch.zeros(capacity).cuda()
#         self.new_obs_mem = torch.zeros(capacity, *obs_dim).cuda() if obs_dim[0] > 1 else torch.zeros(capacity).cuda()
#         self.action_mem = torch.zeros(capacity, *act_dim).cuda() if act_dim[0] > 1 else torch.zeros(capacity).cuda()
#         self.reward_mem = torch.zeros(capacity).cuda()
#         self.done_mem = torch.zeros(capacity).cuda()
#         self.log_prob_mem = torch.zeros(capacity, *act_dim).cuda() if act_dim[0] > 1 else torch.zeros(capacity).cuda()
#         self.advantage_mem = torch.zeros(capacity).cuda()
#         self.value_mem = torch.zeros(capacity).cuda()
#         self.epochs_len = torch.zeros(capacity).cuda()
#
#     def reset(self):
#         self.__init__(self.capacity, self.env_name)
#
#     def append(self, old_obs_batch, new_obs_batch, action_batch, reward_batch, done_batch,
#                log_prob_batch, advantage_batch, value_batch):
#         # insert segment boundary
#         batch_size = len(old_obs_batch)
#         start, end = self.offset, self.offset + batch_size
#         # insert batches to memories
#         self.old_obs_mem[start: end] = old_obs_batch[:]
#         self.new_obs_mem[start: end] = new_obs_batch[:]
#         self.action_mem[start: end] = action_batch[:]
#         self.reward_mem[start: end] = reward_batch[:]
#         self.done_mem[start: end] = done_batch[:]
#         self.log_prob_mem[start: end] = log_prob_batch[:]
#         self.advantage_mem[start: end] = advantage_batch[:]
#         self.value_mem[start: end] = value_batch[:]
#         self.offset += batch_size
#
#     def sample(self, batch_size):
#         out = np.random.choice(self.offset, batch_size, replace=False)
#         return (
#             self.old_obs_mem[out],
#             self.new_obs_mem[out],
#             self.action_mem[out],
#             self.reward_mem[out],
#             self.done_mem[out],
#             self.log_prob_mem[out],
#             self.advantage_mem[out],
#             self.value_mem[out],
#         )
#
#     def compute_epoch_length(self):
#         done_indexes = (self.done_mem == 1).nonzero()
#         self.epochs_len = torch.Tensor([done_indexes[i] - done_indexes[i - 1] if i > 0 else done_indexes[0]
#                                         for i in reversed(range(len(done_indexes)))])
#         return self.epochs_len


@ray.remote
def gen_env(env_name):
    return gym.make(env_name)


@ray.remote
def rollout_single_step_parallel(task_id, env_name, actor, horizon, env=1):
    time_1 = time.time()
    env = gym.make(env_name)
    obs = env.reset()
    total_reward = 0.
    for _ in range(horizon):
        obs, reward, _, _ = env.step(actor.gen_action(torch.Tensor(obs))[0])
        total_reward += reward
    time_2 = time.time()
    print(''.format(time_2 - time_1))
    print("id={}, reward: {}, episode_time: {:.3f}sec".format(task_id, total_reward, time_2 - time_1))
    return total_reward


def rollout_single_step(task_id, env, actor, horizon):
    time_1 = time.time()
    # env = gym.make(env_name)
    # envs = [gym.make(env_name) for _ in range(50)]
    # env = ray.get(gen_env.remote(env_name))
    obs = env.reset()
    total_reward = 0.
    for _ in range(horizon):
        obs, reward, _, _ = env.step(actor.gen_action(torch.Tensor(obs))[0])
        total_reward += reward
    time_2 = time.time()
    print("id={}, reward: {}, episode_time: {:.3f}sec".format(task_id, total_reward, time_2 - time_1))
    return total_reward


def parallel_rollout(env_name, env_number, horizon):
    envs = [gym.make(env_name) for _ in range(env_number)]
    actors = [Actor(4, 1, 32, 3) for _ in range(env_number)]
    time_start = time.time()
    data = ray.get(
        [rollout_single_step_parallel.remote(i, env_name, actors[i], horizon) for i in range(env_number)])
    time_end = time.time()
    print("parallel_time: {}, data:{}".format(time_end - time_start, data))


def serial_rollout(env_name, env_number, horizon):
    envs = [gym.make(env_name) for _ in range(env_number)]
    actors = [Actor(4, 1, 32, 3) for _ in range(env_number)]
    time_start = time.time()
    data = [rollout_single_step(i, envs[i], actors[i], horizon) for i in range(env_number)]
    time_end = time.time()
    print("parallel_time: {}, data:{}".format(time_end - time_start, data))


def loop_rollout(env_name, env_number, horizon):
    envs = [gym.make(env_name) for _ in range(env_number)]
    actor = Actor(4, 1, 32, 3)
    time_start = time.time()
    for env in envs:
        time_1 = time.time()
        obs = env.reset()
        total_reward = 0.
        for _ in range(horizon):
            obs, reward, _, _ = env.step(actor.gen_action(torch.Tensor(obs))[0])
            total_reward += reward
        time_2 = time.time()
        print('episode_time={}'.format(time_2 - time_1))
    time_end = time.time()
    print("parallel_time: {}".format(time_end - time_start))


"""
Test SubprocVecEnv
"""


def make_env(env_id, seed):
    def _f():
        env = gym.make(env_id)
        env.seed(seed)
        return env

    return _f


def subprocenv_rollout(env_name, env_number, horizon):
    time_start = time.time()
    envs = [make_env(env_name, seed) for seed in range(env_number)]
    envs = SubprocVecEnv(envs)
    obs = envs.reset()
    for t in range(horizon):
        action = np.stack([envs.action_space.sample() for _ in range(env_number)])
        obs, reward, done, info = envs.step(action)
    time_end = time.time()
    print("parallel_time: {}".format(time_end - time_start))


"""
Minimal Simulation for Proj
"""


@ray.remote
def rollout_sim_single_step_parallel(task_id, env_name, actor, horizon):
    time_1 = time.time()
    # initialize environment
    env = gym.make(env_name)
    env.seed(task_id)
    # initialize logger
    old_states, new_states, raw_actions, dones, rewards, log_probs, advantages, episode_reward = [], [], [], [], [], [], [], 0.
    # collect episode
    old_obs = env.reset()
    for step in range(horizon):
        # interact with environment
        action, log_prob, raw_action = actor.gen_action(torch.Tensor(old_obs))
        new_obs, reward, done, info = env.step(action)
        # record trajectory step
        old_states.append(old_obs)
        new_states.append(new_obs)
        raw_actions.append(raw_action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)
        episode_reward += reward
        # update old observation
        old_obs = new_obs
        # if done:
        #     break
    dones[-1] = True
    time_2 = time.time()
    print("    id={}, reward: {}, episode_time: {:.3f}sec".format(task_id, episode_reward, time_2 - time_1))
    return [old_states, new_states, raw_actions, rewards, dones, log_probs, episode_reward]


def parallel_rollout_sim(env_name, env_number, horizon):
    actors = [Actor(4, 1, 32, 3) for _ in range(env_number)]
    critic = CriticNet(4, 32)
    rolloutmem = RolloutMemory(env_number * horizon, env_name)
    episodes_rewards = []
    time_start = time.time()
    data = ray.get(
        [rollout_sim_single_step_parallel.remote(i, env_name, actors[i], horizon) for i in range(env_number)])
    time_end = time.time()
    for episode in data:
        old_states, new_states, raw_actions, rewards, dones, log_probs, episode_reward = \
            torch.Tensor(episode[0]), torch.Tensor(episode[1]), torch.Tensor(episode[2]), \
            torch.Tensor(episode[3]), torch.Tensor(episode[4]), torch.Tensor(episode[5]), episode[6]
        gae_deltas = critic.gae_delta(old_states, new_states, rewards, 0.99)
        advantages = torch.Tensor(get_advantage_new(gae_deltas, 0.99, 0.95))
        values = get_values(rewards, 0.99)
        rolloutmem.append(old_states, new_states, raw_actions, rewards, dones, log_probs, advantages, values)
        episodes_rewards.append(episode_reward)
    print("parallel_time: {}\ndata_len: {}\navgR: {:.3f}\nsaved_step_num: {}\n\n"
          .format(time_end - time_start, len(data), torch.mean(torch.Tensor(episodes_rewards)), rolloutmem.offset))
    return torch.mean(torch.Tensor(episodes_rewards)), time_end - time_start


def serial_rollout_sim(env_name, env_number, horizon):
    actor = Actor(4, 1, 32, 3)
    envs = [gym.make(env_name) for _ in range(env_number)]
    for i in range(env_number): envs[i].seed(seed=i)
    data = []
    time_start = time.time()
    for env_id in range(len(envs)):
        env = envs[env_id]
        time_1 = time.time()
        # initialize logger
        old_states, new_states, raw_actions, dones, rewards, log_probs, advantages, episode_reward = \
            [], [], [], [], [], [], [], 0.
        # collect episode
        old_obs = env.reset()
        for step in range(horizon):
            # interact with environment
            action, log_prob, raw_action = actor.gen_action(torch.Tensor(old_obs))
            new_obs, reward, done, info = env.step(action)
            # record trajectory step
            old_states.append(old_obs)
            new_states.append(new_obs)
            raw_actions.append(raw_action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            episode_reward += reward
            # update old observation
            old_obs = new_obs
            # if done:
            #     break
        dones[-1] = True
        time_2 = time.time()
        data.append([old_states, new_states, raw_actions, rewards, dones, log_probs, episode_reward])
        print("    env_id={}, reward: {}, episode_time: {:.3f}sec".format(env_id, episode_reward, time_2 - time_1))
    time_end = time.time()
    print("parallel_time: {}\ndata_len:{}\n\n".format(time_end - time_start, len(data)))


if __name__ == "__main__":
    # ray.init(log_to_driver=False)
    ray.init()
    # parallel_work(50)
    # serial_work(10)
    # parallel_rollout('InvertedPendulum-v2', 50, 80)
    # serial_rollout('InvertedPendulum-v2', 50, 80)
    # loop_rollout('InvertedPendulum-v2', 50, 80)
    # subprocenv_rollout('InvertedPendulum-v2', 50, 10)
    parallel_rollout_sim('InvertedPendulum-v2', 50, 1000)
    # serial_rollout_sim('InvertedPendulum-v2', 100, 160)
