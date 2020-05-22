import torch
from torch.utils.tensorboard import SummaryWriter
import time
import ray
import copy
import gym
import multiprocessing as mp
import threading

from utils import gen_env, gen_actor, gen_critic, count_model_params, get_advantage, get_advantage_new, get_values, \
    get_entropy, log_policy_rollout, AverageMeter
from rolloutmemory import RolloutMemory
from networks import get_norm_log_prob


@ray.remote(num_gpus=3)
def rollout_sim_single_step_parallel(task_id, env, actor, horizon):
    # time_1 = time.time()
    # initialize logger
    old_states, new_states, raw_actions, dones, rewards, log_probs, advantages, episode_reward = [], [], [], [], [], [], [], 0.
    # collect episode
    old_obs = env.reset()
    for step in range(horizon):
        # interact with environment
        action, log_prob, raw_action = actor.gen_action(torch.Tensor(torch.Tensor(old_obs)).cuda())
        new_obs, reward, done, info = env.step(action.cpu())
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
        if done:
            break
    dones[-1] = True
    # time_2 = time.time()
    # print("    id={}, reward: {}, episode_time: {:.3f}sec".format(task_id, episode_reward, time_2 - time_1))
    return [old_states, new_states, raw_actions, rewards, dones, log_probs, episode_reward]


def rollout_serial(rolloutmem, envs, actor, critic, params):
    episodes_rewards = []
    # collect episodes from different environments
    for env in envs:
        old_states, new_states, raw_actions, dones, rewards, log_probs, advantages, episode_reward \
            = [], [], [], [], [], [], [], 0.
        # collect one episode from current env
        old_state = env.reset()
        for step in range(params.policy_params.horizon):
            # act one step in current environment
            action, log_prob, raw_action = actor.gen_action(torch.Tensor(old_state).cuda())
            new_state, reward, done, info = env.step(action.cpu())
            time.sleep(.002)  # check this issue: https://github.com/openai/mujoco-py/issues/340
            # record trajectory step
            old_states.append(old_state)
            new_states.append(new_state)
            raw_actions.append(raw_action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            episode_reward += reward
            # update old observation
            old_state = new_state
            if done:
                break
        dones[-1] = True
        # reformat trajectory step
        old_states, new_states, raw_actions, rewards, dones, log_probs = \
            torch.Tensor(old_states).cuda(), torch.Tensor(new_states).cuda(), torch.stack(raw_actions).detach().cuda(), \
            torch.Tensor(rewards).cuda(), torch.Tensor(dones).cuda(), torch.stack(log_probs).detach().cuda()
        # compute loss factors
        gae_deltas = critic.gae_delta(old_states, new_states, rewards, params.policy_params.discount)
        for t in range(len(gae_deltas)):
            advantages.append(get_advantage(t, gae_deltas, params.policy_params.discount, params.policy_params.lambd))
        advantages = torch.Tensor(advantages).cuda()
        values = get_values(rewards, params.policy_params.discount).cuda()
        if len(advantages.shape) == 1: advantages = advantages[:, None]
        if len(values.shape) == 1: values = values[:, None]
        # store epoch
        rolloutmem.append(old_states, new_states, raw_actions, rewards, dones, log_probs, advantages, values)
        # record epoch reward
        episodes_rewards.append(episode_reward)
    return torch.mean(torch.Tensor(episodes_rewards))


def rollout_parallel(rolloutmem, envs, actor, critic, params):
    episodes_rewards = []
    data = ray.get(
        [rollout_sim_single_step_parallel.remote(i, envs[i], copy.deepcopy(actor), params.policy_params.horizon)
         for i in range(params.policy_params.envs_num)])
    for episode in data:
        old_states, new_states, raw_actions, rewards, dones, log_probs, episode_reward = \
            torch.Tensor(episode[0]).cuda(), torch.Tensor(episode[1]).cuda(), torch.stack(episode[2]).detach().cuda(), \
            torch.Tensor(episode[3]).cuda(), torch.Tensor(episode[4]).cuda(), torch.stack(episode[5]).detach().cuda(), \
            torch.Tensor([episode[6]]).cuda()
        gae_deltas = critic.gae_delta(old_states, new_states, rewards, params.policy_params.discount)
        advantages = torch.Tensor(
            get_advantage_new(gae_deltas, params.policy_params.discount, params.policy_params.lambd)).cuda()
        values = get_values(rewards, params.policy_params.discount).cuda()
        if len(advantages.shape) == 1: advantages = advantages[:, None]
        if len(values.shape) == 1: values = values[:, None]
        rolloutmem.append(old_states, new_states, raw_actions, rewards, dones, log_probs, advantages, values)
        episodes_rewards.append(episode_reward)
    return torch.mean(torch.Tensor(episodes_rewards))


parallel_rollout = False


def rollout(rolloutmem, envs, actor, critic, params, iteration):
    # select bets rollout method during training
    # global parallel_rollout
    # if not parallel_rollout:
    #     if iteration % params.parallel_check_num == 0:
    #         t_1 = time.time()
    #         rollout_parallel(rolloutmem, envs, actor, critic, params)
    #         rolloutmem.reset()
    #         t_2 = time.time()
    #         mean_episode_reward = rollout_serial(rolloutmem, envs, actor, critic, params)
    #         t_3 = time.time()
    #         time_p, time_s = t_2 - t_1, t_3 - t_2
    #         if time_s - time_p > 2.:
    #             parallel_rollout = True
    #             print(
    #                 "\n\n    >>> parallel_time: {:.3f}, serial_time: {:.3f}, change to parallel rollout.<<<\n\n".format(
    #                     time_p, time_s))
    #         else:
    #             print("\n    >>> checking parallel priority: parallel_time: {:.3f}, serial_time: {:.3f}<<<\n".format(
    #                 time_p, time_s))
    #     else:
    #         mean_episode_reward = rollout_serial(rolloutmem, envs, actor, critic, params)
    # else:
    #     mean_episode_reward = rollout_parallel(rolloutmem, envs, actor, critic, params)
    if params.parallel:
        mean_episode_reward = rollout_parallel(rolloutmem, envs, actor, critic, params)
    else:
        mean_episode_reward = rollout_serial(rolloutmem, envs, actor, critic, params)
    return mean_episode_reward


def optimize_step(optimizer, rolloutmem, actor, critic, params, iteration):
    entropy_discount = 1.
    if params.reducing_entro_loss:
        entropy_discount = 1. - iteration / params.iter_num
    # sample rollout steps from current policy
    old_obs_batch, _, raw_action_batch, reward_batch, done_batch, old_log_prob_batch, advantage_batch, value_batch \
        = rolloutmem.sample(params.policy_params.batch_size)
    # compute loss factors
    mean, cov = actor.policy_out(old_obs_batch)
    new_log_prob_batch = get_norm_log_prob([mean, cov], raw_action_batch, actor.scale)
    ratio = torch.exp(new_log_prob_batch - old_log_prob_batch)
    surr1 = ratio * advantage_batch
    surr2 = torch.clamp(ratio, 1 - params.policy_params.clip_param,
                        1 + params.policy_params.clip_param) * advantage_batch
    # compute losses
    policy_loss = - torch.mean(torch.min(surr1, surr2))
    critic_loss = torch.mean(torch.pow(critic.forward(old_obs_batch) - value_batch, 2))  # MSE loss
    entropy_loss = - torch.mean(get_entropy([mean, cov]))
    loss = policy_loss \
           + params.policy_params.critic_coef * critic_loss \
           + params.policy_params.entropy_coef * entropy_discount * entropy_loss
    # gradient descent
    optimizer.zero_grad()
    loss.backward()
    for param in list(actor.parameters()) + list(critic.parameters()):
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    # return variables for logger
    return loss, policy_loss, critic_loss, entropy_loss, advantage_batch, ratio, surr1, surr2, torch.mean(
        rolloutmem.compute_epoch_length())


def train(params):
    # algorithm ingredients instantiation
    torch.manual_seed(params.seed)
    actor = gen_actor(params.env_name, params.policy_params.hidden_dim)
    critic = gen_critic(params.env_name, params.policy_params.hidden_dim)
    rolloutmem = RolloutMemory(params.policy_params.envs_num * params.policy_params.horizon, params.env_name)
    envs = [gen_env(params.env_name) for i in range(params.policy_params.envs_num)]
    for i in range(len(envs)): envs[i].seed(seed=params.seed + i)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()),
                                 lr=params.policy_params.learning_rate)
    ray.init(log_to_driver=False, local_mode=False)
    # logger instantiation
    tb = SummaryWriter()
    rollout_time, update_time = AverageMeter(), AverageMeter()
    print("----------------------------------")
    print("Training model with {} parameters...".format(count_model_params(actor) + count_model_params(critic)))
    print("----------------------------------")
    # training loop
    for iteration in range(int(params.iter_num)):
        # collect rollouts from current policy
        rolloutmem.reset()
        iter_start_time = time.time()
        mean_iter_reward = rollout(rolloutmem, envs, actor, critic, params, iteration)
        # optimize by gradient descent
        update_start_time = time.time()
        loss, policy_loss, critic_loss, entropy_loss, advantage, ratio, surr1, surr2, epochs_len = \
            None, None, None, None, None, None, None, None, None,
        for epoch in range(params.policy_params.epochs_num):
            loss, policy_loss, critic_loss, entropy_loss, advantage, ratio, surr1, surr2, epochs_len = \
                optimize_step(optimizer, rolloutmem, actor, critic, params, iteration)
        tb.add_scalar('loss', loss, iteration)
        tb.add_scalar('policy_loss', -1 * policy_loss, iteration)
        tb.add_scalar('critic_loss', critic_loss, iteration)
        tb.add_scalar('entropy_loss', -1 * entropy_loss, iteration)
        tb.add_scalar('advantage', advantage.mean(), iteration)
        tb.add_scalar('ratio', ratio.mean(), iteration)
        tb.add_scalar('surr1', surr1.mean(), iteration)
        tb.add_scalar('surr2', surr2.mean(), iteration)
        tb.add_scalar('epoch_len', epochs_len, iteration)
        tb.add_scalar('rewards', mean_iter_reward, iteration)
        iter_end_time = time.time()
        rollout_time.update(update_start_time - iter_start_time)
        update_time.update(iter_end_time - update_start_time)
        tb.add_scalar('rollout_time', rollout_time.val, iteration)
        tb.add_scalar('update_time', update_time.val, iteration)
        print('it {}: avgR: {:.3f} | rollout_time: {:.3f}sec update_time: {:.3f}sec'
              .format(iteration, mean_iter_reward, rollout_time.val, update_time.val))
        # save rollout video
        if iteration % int(params.plotting_iters) == 0 and iteration > 0:
            log_policy_rollout(params, actor, envs[0], 'iter-{}'.format(iteration))
    # save rollout videos
    for i in range(3):
        log_policy_rollout(params, actor, envs[0], 'final-{}'.format(i))
