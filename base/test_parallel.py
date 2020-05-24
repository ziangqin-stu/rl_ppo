import copy
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
from networks import get_norm_log_prob, CriticFC
from rolloutmemory import RolloutMemory

"""
Test RAY Ability
"""


@ray.remote
class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 1

    def read(self):
        return self.n


def test_ray_actor():
    counters = [Counter.remote() for i in range(4)]
    [c.increment.remote() for c in counters]
    futures = [c.read.remote() for c in counters]
    print(ray.get(futures))


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


@ray.remote
def gen_env(env_name):
    return gym.make(env_name)


@ray.remote
def rollout_single_step_parallel(task_id, env_name, actor, horizon):
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
    actors = [gen_actor(env_name, 512) for _ in range(env_number)]
    time_start = time.time()
    data = ray.get(
        [rollout_single_step_parallel.remote(i, env_name, actors[i], horizon) for i in range(env_number)])
    time_end = time.time()
    print("parallel_time: {}, data:{}".format(time_end - time_start, data))


def serial_rollout(env_name, env_number, horizon):
    envs = [gym.make(env_name) for _ in range(env_number)]
    actors = [gen_actor(env_name, 512) for _ in range(env_number)]
    time_start = time.time()
    data = [rollout_single_step(i, envs[i], actors[i], horizon) for i in range(env_number)]
    time_end = time.time()
    print("parallel_time: {}, data:{}".format(time_end - time_start, data))


def loop_rollout(env_name, env_number, horizon):
    envs = [gym.make(env_name) for _ in range(env_number)]
    actor = gen_actor(env_name, 512)
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


@ray.remote(num_gpus=3)
def rollout_sim_single_step_parallel(task_id, env_name, horizon, actor=None, env=None):
    time_1 = time.time()
    # initialize environment
    if actor is None: actor = gen_actor(env_name, 512)
    if env is None: env = gym.make(env_name)
    # env.seed(task_id)
    # initialize logger
    old_states, new_states, raw_actions, dones, rewards, log_probs, advantages, episode_reward = [], [], [], [], [], [], [], 0.
    # collect episode
    old_obs = env.reset()
    for step in range(horizon):
        # interact with environment
        action, log_prob, raw_action = actor.gen_action(torch.Tensor(old_obs).cuda())
        assert (env.action_space.low < np.array(action)).all() and (np.array(action) < env.action_space.high).all()
        new_obs, reward, done, info = env.step(action)
        # record trajectory step
        old_states.append(old_obs)
        new_states.append(new_obs)
        raw_actions.append(raw_action.view(-1))
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
    envs = [gym.make(env_name) for _ in range(env_number)]
    actor = gen_actor(env_name, 512)
    critic = gen_critic(env_name, 512)
    rolloutmem = RolloutMemory(env_number * horizon, env_name)
    time_start = time.time()
    episodes_rewards = []
    data = ray.get(
        [rollout_sim_single_step_parallel.remote(i, env_name, horizon, None, None) for i in range(env_number)])
    time_end = time.time()
    for episode in data:
        old_states, new_states, raw_actions, rewards, dones, log_probs, episode_reward = \
            torch.Tensor(episode[0]).cuda(), torch.Tensor(episode[1]).cuda(), torch.stack(episode[2]).detach().cuda(), \
            torch.Tensor(episode[3]).cuda(), torch.Tensor(episode[4]).cuda(), torch.stack(episode[5]).detach().cuda(), \
            torch.Tensor([episode[6]]).cuda()
        gae_deltas = critic.gae_delta(old_states, new_states, rewards, 0.99)
        advantages = torch.Tensor(get_advantage_new(gae_deltas, 0.99, 0.95)).cuda()
        values = get_values(rewards, 0.99).cuda()
        if len(advantages.shape) == 1: advantages = advantages[:, None]
        if len(values.shape) == 1: values = values[:, None]
        rolloutmem.append(old_states, new_states, raw_actions, rewards, dones, log_probs, advantages, values)
        episodes_rewards.append(episode_reward)
    time_reformat = time.time()
    print(
        "parallel_time: {}, reformat_time: {:.3f}\nrollout_time: {:.3f}\ndata_len: {}\navgR: {:.3f}\nsaved_step_num: {}\n\n"
            .format(time_end - time_start, time_reformat - time_end, time_reformat - time_start, len(data),
                    torch.mean(torch.Tensor(episodes_rewards)), rolloutmem.offset))
    return torch.mean(torch.Tensor(episodes_rewards)), time_end - time_start


def serial_rollout_sim(env_name, env_number, horizon):
    actor = gen_actor(env_name, 512)
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


"""
Ayush's Method
"""


@ray.remote
class ParallelEnv:
    def __init__(self, env_name, id):
        self.env = gym.make(env_name)
        self.id = id
        # print("env_{} created!".format(self.id))

    def reset(self):
        obs = self.env.reset()
        # print("env_{} reset!".format(self.id))
        return obs

    def step(self, state):
        step_data = self.env.step(state)
        # print("env_{} step!".format(self.id))
        return step_data


def parallel_rollout_env(envs, actor, critic, rolloutmem, horizon):
    # interact
    time_start = time.time()
    env_number = len(envs)
    old_states, new_states, raw_actions, dones, rewards, log_probs, advantages, episode_reward \
        = [], [], [], [], [], [], [], [0] * env_number
    old_state = ray.get([env.reset.remote() for env in envs])
    for step in range(horizon):
        # interact
        action, log_prob, raw_action = actor.gen_action(torch.Tensor(old_state).cuda())
        step_obs_batch = ray.get(
            [envs[i].step.remote(action[i]) for i in range(env_number)])  # new_obs, reward, done, info
        # parse interact results
        new_state = [step_obs[0] for step_obs in step_obs_batch]
        reward = [step_obs[1] for step_obs in step_obs_batch]
        done = [step_obs[2] for step_obs in step_obs_batch]
        # record parsed results
        old_states.append(old_state)
        new_states.append(new_state)
        raw_actions.append(raw_action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)
        episode_reward = [float(reward[i]) + episode_reward[i] for i in range(len(reward))]
        # update old observation
        old_state = new_state
        if np.array(done).all():
            break
    dones[-1] = [True] * env_number
    old_states = torch.Tensor(old_states).permute(1, 0, 2).cuda()
    new_states = torch.Tensor(new_states).permute(1, 0, 2).cuda()
    raw_actions = torch.stack(raw_actions).permute(1, 0, 2).detach().cuda()
    rewards = torch.Tensor(rewards).permute(1, 0).cuda()
    dones = torch.Tensor(dones).permute(1, 0).cuda()
    log_probs = torch.stack(log_probs).permute(1, 0, 2).detach().cuda()
    gae_deltas = critic.gae_delta(old_states, new_states, rewards, .99).cuda()
    advantages = torch.Tensor(get_advantage_new(gae_deltas, .99, .95)).cuda()
    values = get_values(rewards, .99).cuda()
    advantages = advantages[:, :, None]
    values = values[:, :, None]
    # rolloutmem.append(old_states, new_states, raw_actions, rewards, dones, log_probs, advantages, values)
    for i in range(env_number):
        # abandon redundant step info
        first_done = (dones[i] > 0).nonzero().min()
        rolloutmem.append(old_states[i][:first_done + 1], new_states[i][:first_done + 1],
                          raw_actions[i][:first_done + 1], rewards[i][:first_done + 1], dones[i][:first_done + 1],
                          log_probs[i][:first_done + 1], advantages[i][:first_done + 1], values[i][:first_done + 1])
    print("    rollout_time: {}".format(time.time() - time_start))
    return torch.mean(torch.Tensor(episode_reward))


def repeat_rollout(env_name, env_number, horizon, iter_num):
    # ingredients prepare
    time_start = time.time()
    envs = [ParallelEnv.remote(env_name, id) for id in range(env_number)]
    actor = gen_actor(env_name, 512)
    critic = gen_critic(env_name, 512)
    rolloutmem = RolloutMemory(env_number * horizon, env_name)
    print("    build_time: {}".format(time.time() - time_start))
    # repeat iteration
    for i in range(iter_num):
        print("iter_{}".format(i))
        parallel_rollout_env(envs, actor, critic, rolloutmem, horizon)
    print("Work Done!")


if __name__ == "__main__":
    # ray.init(log_to_driver=False)
    # ray.init(logging_level='ERROR')
    ray.init(log_to_driver=False)
    # parallel_work(50)
    # serial_work(10)
    # parallel_rollout('InvertedPendulum-v2', 50, 80)
    # serial_rollout('InvertedPendulum-v2', 50, 80)
    # loop_rollout('InvertedPendulum-v2', 50, 80)
    # subprocenv_rollout('InvertedPendulum-v2', 50, 10)
    # parallel_rollout_sim('Hopper-v2', 50, 200)
    repeat_rollout('Hopper-v2', 5, 20, 5)
    # test_paraenv('Hopper-v2')
    # test_ray_actor()
    # parallel_rollout_sim('InvertedDoublePendulum-v2', 50, 200)
    # serial_rollout_sim('InvertedPendulum-v2', 100, 160)
