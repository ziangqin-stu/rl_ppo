import copy
import os
import time
import random
import gc

import numpy as np
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from data import envnames_minigrid
from networks import get_norm_log_prob
from rolloutmemory import RolloutMemory
from utils import gen_actor, gen_critic, get_dist_type, count_model_params, get_advantage, get_advantage_new, \
    get_values, get_entropy, log_policy_rollout, logger_scalar, logger_histogram, save_model, test_rollout, ParallelEnv, \
    AverageMeter


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
            assert (env.action_space.low < np.array(action)).all() and (np.array(action) < env.action_space.high).all()
            new_state, reward, done, info = env.step(action.cpu() if hasattr(action, 'cpu') else action)
            time.sleep(.002)  # check this issue: https://github.com/openai/mujoco-py/issues/340
            # record trajectory step
            old_states.append(old_state)
            new_states.append(new_state)
            raw_actions.append(raw_action.view(-1))
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.view(-1))
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


@ray.remote(num_gpus=1)
def rollout_sim_single_step_parallel(task_id, env, actor, horizon):
    # initialize logger
    old_states, new_states, raw_actions, dones, rewards, log_probs, advantages, rollout_reward = [], [], [], [], [], [], [], 0.
    # collect episode
    old_obs = ray.get(env.reset.remote())
    env_attributes = ray.get(env.get_attributes.remote())
    for step in range(horizon):
        # interact with environment
        action, log_prob, raw_action = actor.gen_action(torch.Tensor(old_obs).cuda())
        assert (env_attributes['action_high'].cuda() >= action.cuda()).all() and (
                action.cuda() >= env_attributes['action_low'].cuda()).all(), '>> Error: action value exceeds boundary!'
        [new_obs, reward, done, info] = ray.get(env.step.remote(action))
        # record trajectory step
        old_states.append(old_obs)
        new_states.append(new_obs)
        raw_actions.append(raw_action.view(-1))
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob.view(-1))
        rollout_reward += reward
        # update old observation
        old_obs = new_obs
        if done:
            old_obs = ray.get(env.reset.remote())
    dones[-1] = True
    return [old_states, new_states, raw_actions, rewards, dones, log_probs, rollout_reward]


def rollout_parallel(rolloutmem, envs, actor, critic, params):
    # parallelization method_1
    episodes_rewards = []
    episode_number = []
    data = ray.get(
        [rollout_sim_single_step_parallel.remote(i, envs[i], copy.deepcopy(actor), params.policy_params.horizon)
         for i in range(params.policy_params.envs_num)])
    for episode in data:
        old_states, new_states, raw_actions, rewards, dones, log_probs, rollout_reward = \
            torch.Tensor(episode[0]).cuda(), torch.Tensor(episode[1]).cuda(), torch.stack(episode[2]).detach().cuda(), \
            torch.Tensor(episode[3]).cuda(), torch.Tensor(episode[4]).cuda(), torch.stack(episode[5]).detach().cuda(), \
            torch.Tensor([episode[6]]).cuda()
        gae_deltas = critic.gae_delta(old_states, new_states, rewards, params.policy_params.discount)
        advantages = get_advantage_new(gae_deltas, params.policy_params.discount,
                                       params.policy_params.lambd).detach().cuda()
        values = get_values(rewards, params.policy_params.discount).cuda()
        if len(advantages.shape) == 1: advantages = advantages[:, None]
        if len(values.shape) == 1: values = values[:, None]
        rolloutmem.append(old_states, new_states, raw_actions, rewards, dones, log_probs, advantages, values)
        episodes_rewards.append(rollout_reward)
        episode_number.append(len((dones == 1).nonzero()))
    return torch.mean(torch.Tensor([episodes_rewards[i] / max(episode_number[i], 1) for i in range(len(envs))]))


def parallel_rollout_env(rolloutmem, envs, actor, critic, params):
    # parallelization method_2
    # interact
    env_number = len(envs)
    env_attributes = ray.get(envs[0].get_attributes.remote())
    old_states, new_states, raw_actions, dones, rewards, log_probs, advantages, rollout_reward, episode_number \
        = [], [], [], [], [], [], [], [0] * env_number, [0] * env_number
    old_state = ray.get([env.reset.remote() for env in envs])
    rolloutmem.reset()
    for step in range(params.policy_params.horizon):  # data shape: [env_num, data[:, ..., step_i]]
        # interact
        action, log_prob, raw_action = actor.gen_action(torch.Tensor(old_state).cuda())
        action = action.cuda()
        assert (env_attributes['action_high'].cuda() >= action).all() and (
                action >= env_attributes['action_low'].cuda()).all(), '>> Error: action value exceeds boundary!'
        action = action.cpu()
        if env_attributes['action_type']['data_type'] is type(int(0)) and not env_attributes['image_obs']:
            action = action.int().tolist()
        step_obs_batch = ray.get(
            [envs[i].step.remote(action[i]) for i in range(env_number)])  # new_obs, reward, done, info
        # parse interact results
        new_state = [step_obs[0] for step_obs in step_obs_batch]
        reward = [step_obs[1] for step_obs in step_obs_batch]
        done = [step_obs[2] for step_obs in step_obs_batch]
        # record parsed results
        raw_action = raw_action[:, None] if len(raw_action.shape) < 2 else raw_action
        old_states.append(old_state)
        new_states.append(new_state)
        raw_actions.append(raw_action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob.float())
        rollout_reward = [rollout_reward[i] + (float(reward[i]) if done[i] is False else 0) for i in range(len(reward))]
        episode_number = [float(done[i]) + episode_number[i] for i in range(len(done))]
        # update old observation
        old_state = new_state
        if torch.Tensor(done).bool().all():
            break
        # for ind in [int(i) for i in list((torch.Tensor(done) == 1).nonzero())]:
        #     state = ray.get(envs[ind].reset.remote())
    dones[-1] = [True] * env_number
    # reformat collected data to episode-serial order
    if env_attributes['image_obs']:
        old_states = torch.Tensor(old_states).permute(1, 0, 2, 3, 4).cuda()
        new_states = torch.Tensor(new_states).permute(1, 0, 2, 3, 4).cuda()
        raw_actions = torch.stack(raw_actions).permute(1, 0, 2).detach().cuda()
        rewards = torch.Tensor(rewards).permute(1, 0).cuda()
        dones = torch.Tensor(dones).permute(1, 0).cuda()
        log_probs = torch.stack(log_probs).permute(1, 0, 2).detach().double().cuda()
    else:
        old_states = torch.Tensor(old_states).permute(1, 0, 2).cuda()
        new_states = torch.Tensor(new_states).permute(1, 0, 2).cuda()
        raw_actions = torch.stack(raw_actions).permute(1, 0, 2).detach().cuda()
        rewards = torch.Tensor(rewards).permute(1, 0).cuda()
        dones = torch.Tensor(dones).permute(1, 0).cuda()
        log_probs = torch.stack(log_probs).permute(1, 0, 2).detach().double().cuda()
    for i in range(env_number):
        # compute each episode length
        first_done = (dones[i] > 0).nonzero().min()
        gae_deltas = critic.gae_delta(old_states[i][:first_done + 1], new_states[i][:first_done + 1],
                                      rewards[i][:first_done + 1], params.policy_params.discount)
        advantages = get_advantage_new(gae_deltas, params.policy_params.discount, params.policy_params.lambd)[:,
                     None].detach().cuda()
        advantages = advantages[:first_done + 1]
        advantages = (advantages - advantages.mean()) / torch.std(advantages + 1e-6)
        values = get_values(rewards[i][:first_done + 1], params.policy_params.discount)[:, None].cuda()
        # abandon redundant step info

        rolloutmem.append(old_states[i][:first_done + 1], new_states[i][:first_done + 1],
                          raw_actions[i][:first_done + 1], rewards[i][:first_done + 1], dones[i][:first_done + 1],
                          log_probs[i][:first_done + 1], advantages[:first_done + 1], values[:first_done + 1])
        # rolloutmem.append(old_states[i], new_states[i], raw_actions[i], rewards[i], dones[i], log_probs[i], advantages,
        #                   values)

    # if env_attributes['final_reward']:
    #     return torch.mean(torch.Tensor(rollout_reward)) / max(torch.mean(torch.Tensor(episode_number)), torch.Tensor([1.]))
    # else:
    #     return torch.mean(torch.Tensor(rollout_reward))
    # return torch.mean(torch.Tensor(rollout_reward)) / max(torch.mean(torch.Tensor(episode_number)), torch.Tensor([1.]))
    return torch.mean(torch.Tensor(rollout_reward))


def rollout(rolloutmem, envs, actor, critic, params):
    if params.parallel:
        # mean_rollout_reward = rollout_parallel(rolloutmem, envs, actor, critic, params)
        mean_rollout_reward = parallel_rollout_env(rolloutmem, envs, actor, critic, params)
    else:
        mean_rollout_reward = rollout_serial(rolloutmem, envs, actor, critic, params)
    return float(mean_rollout_reward)


def optimize_step(optimizer, rolloutmem, actor, critic, params, iteration):
    entropy_discount = 1.
    if params.decay_entro_loss:
        entropy_discount = 1. - iteration / params.iter_num
    # print('entropy_discount: {}'.format(entropy_discount))
    # sample rollout steps from current policy
    old_obs_batch, _, raw_action_batch, reward_batch, done_batch, old_log_prob_batch, advantage_batch, value_batch \
        = rolloutmem.sample(params.policy_params.batch_size)
    # compute loss factors
    logits = actor.policy_out(old_obs_batch)
    new_log_prob_batch = get_norm_log_prob(logits, raw_action_batch, actor.scale,
                                           dist_type=get_dist_type(params.env_name))
    assert len(new_log_prob_batch.shape) > 1, '    >>> [optimize_step -> new_log_prob_batch], wrong dimension'
    ratio = torch.exp(new_log_prob_batch.double() - old_log_prob_batch.double())
    # if not torch.stack([ratio[i] != float('inf') for i in range(len(ratio))]).bool().all():
    #     print("    >>> inf in ratio")
    # if not torch.stack([ratio[i] < 100 for i in range(len(ratio))]).bool().all():
    #     print("    >>> clipped ratio value!")
    if not torch.stack([~torch.isnan(ratio[i]) for i in range(len(ratio))]).bool().all():
        print("    >>> nan in ratio, clipped ratio value.")
        ratio = torch.clamp(ratio, 0., 3.0e38)
    if ratio.mean() < 0.1:
        print("    >>> low ratio!")

    # assert torch.stack([ratio[i] <= 1.7e308 for i in range(len(ratio))]).bool().all(), \
    #     '    >>> [optimize_step -> ratio], ratio data overflow.'
    surr1 = ratio * advantage_batch.double()
    surr2 = torch.clamp(ratio, 1 - params.policy_params.clip_param,
                        1 + params.policy_params.clip_param) * advantage_batch.double()
    # compute losses
    policy_loss = - torch.mean(torch.min(surr1, surr2)).double()
    critic_loss = torch.mean(torch.pow(critic.forward(old_obs_batch) - value_batch, 2)).double()  # MSE loss
    entropy_loss = - torch.mean(get_entropy(logits, get_dist_type(params.env_name))).double()
    loss = policy_loss \
           + params.policy_params.critic_coef * critic_loss \
           + params.policy_params.entropy_coef * entropy_discount * entropy_loss
    # gradient descent
    optimizer.zero_grad()
    loss.backward()
    # clamp gradient to avoid explosion
    for param in list(actor.parameters()) + list(critic.parameters()):
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    # return variables for logger
    return loss, policy_loss, critic_loss, entropy_loss, advantage_batch, ratio, surr1, surr2, torch.mean(
        rolloutmem.compute_epoch_length())


def train(params):
    # ============
    # Preparations
    # ============
    gc.collect()
    ray.init(log_to_driver=False, local_mode=False, num_gpus=1)  # or, ray.init()
    if not params.use_pretrain:
        # >> algorithm ingredients instantiation
        seed = params.seed
        actor = gen_actor(params.env_name, params.policy_params.hidden_dim)
        critic = gen_critic(params.env_name, params.policy_params.hidden_dim)
        optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()),
                                     lr=params.policy_params.learning_rate)
        rollout_time, update_time = AverageMeter(), AverageMeter()
        iteration_pretrain = 0
        # >> set random seed (for reproducing experiment)
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        # build models
        actor = gen_actor(params.env_name, params.policy_params.hidden_dim).cuda()
        critic = gen_critic(params.env_name, params.policy_params.hidden_dim).cuda()
        optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=0.0001)
        # load models
        print("\n\nLoading training checkpoint...")
        print("------------------------------")
        load_path = os.path.join('./save/model', params.pretrain_file)
        checkpoint = torch.load(load_path)
        seed = checkpoint['seed']
        actor.load_state_dict(checkpoint['actor_state_dict'])
        actor.train()
        critic.load_state_dict(checkpoint['critic_state_dict'])
        critic.train()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        [rollout_time, update_time] = checkpoint['time_recorder']
        iteration_pretrain = checkpoint['iteration']
        # >> set random seed (for reproducing experiment)
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        print("Loading finished!")
        print("------------------------------\n\n")
    rolloutmem = RolloutMemory(params.policy_params.envs_num * params.policy_params.horizon, params.env_name)
    envs = [ParallelEnv.remote(params.env_name, i) for i in range(params.policy_params.envs_num)]
    for i in range(len(envs)): envs[i].seed.remote(seed=seed + i)
    tb = SummaryWriter()
    # ============
    # Training
    # ============
    # >> training loop
    print("----------------------------------")
    print("Training model with {} parameters...".format(count_model_params(actor) + count_model_params(critic)))
    print("----------------------------------")
    time_start = time.time()
    for iteration in range(int(params.iter_num - iteration_pretrain)):
        # collect rollouts from current policy
        rolloutmem.reset()
        iter_start_time = time.time()
        mean_iter_reward = rollout(rolloutmem, envs, actor, critic, params)
        # optimize by gradient descent
        update_start_time = time.time()
        loss, policy_loss, critic_loss, entropy_loss, advantage, ratio, surr1, surr2, epochs_len = \
            None, None, None, None, None, None, None, None, None,
        for epoch in range(params.policy_params.epochs_num):
            loss, policy_loss, critic_loss, entropy_loss, advantage, ratio, surr1, surr2, epochs_len = \
                optimize_step(optimizer, rolloutmem, actor, critic, params, iteration)
        iter_end_time = time.time()
        tb = logger_scalar(tb, iteration + iteration_pretrain, loss, policy_loss, critic_loss, entropy_loss, advantage,
                           ratio, surr1, surr2, epochs_len, mean_iter_reward, time_start)
        # tb = logger_histogram(tb, iteration + iteration_pretrain, actor, critic)
        rollout_time.update(update_start_time - iter_start_time)
        update_time.update(iter_end_time - update_start_time)
        tb.add_scalar('rollout_time', rollout_time.val, iteration + iteration_pretrain)
        tb.add_scalar('update_time', update_time.val, iteration + iteration_pretrain)
        print('it {}: avgR: {:.3f} avgL: {:.3f} | rollout_time: {:.3f}sec update_time: {:.3f}sec'
              .format(iteration + iteration_pretrain, mean_iter_reward, epochs_len, rollout_time.val, update_time.val))
        # save rollout video
        # if (iteration + 1) % int(params.plotting_iters) == 0 and iteration > 0 and params.log_video:
            # log_policy_rollout(params, actor, params.env_name, 'iter-{}'.format(iteration + iteration_pretrain))
        # save model
        if (iteration + 1) % int(params.checkpoint_iter) == 0 and iteration > 0 and params.save_checkpoint:
            save_model(params.prefix, iteration, iteration_pretrain, seed, actor, critic, optimizer, rollout_time,
                       update_time)
    # >> save rollout videos
    if params.log_video:
        save_model(params.prefix, iteration, iteration_pretrain, seed, actor, critic, optimizer, rollout_time,
                   update_time)
        # for i in range(3):
            # log_policy_rollout(params, actor, params.env_name, 'final-{}'.format(i))
