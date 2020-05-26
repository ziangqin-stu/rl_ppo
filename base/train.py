import copy
import os
import time
import random

import numpy as np
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from networks import get_norm_log_prob
from rolloutmemory import RolloutMemory
from utils import gen_env, gen_actor, gen_critic, get_dist_type, count_model_params, get_advantage, get_advantage_new, \
    get_values, get_entropy, log_policy_rollout, ParallelEnv, AverageMeter


@ray.remote(num_gpus=3)
def rollout_sim_single_step_parallel(task_id, env, actor, horizon):
    # initialize logger
    old_states, new_states, raw_actions, dones, rewards, log_probs, advantages, episode_reward = [], [], [], [], [], [], [], 0.
    # collect episode
    old_obs = ray.get(env.reset.remote())
    env_attributes = ray.get(env.get_attributes.remote())
    for step in range(horizon):
        # interact with environment
        action, log_prob, raw_action = actor.gen_action(torch.Tensor(old_obs).cuda())
        assert (env_attributes['action_high'].cuda() >= action).all() and (action >= env_attributes['action_low'].cuda()).all(), '>> Error: action value exceeds boundary!'
        [new_obs, reward, done, info] = ray.get(env.step.remote(action))
        # record trajectory step
        old_states.append(old_obs)
        new_states.append(new_obs)
        raw_actions.append(raw_action.view(-1))
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob.view(-1))
        episode_reward += reward
        # update old observation
        old_obs = new_obs
        if done:
            break
    dones[-1] = True
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
        advantages = get_advantage_new(gae_deltas, params.policy_params.discount, params.policy_params.lambd).detach().cuda()
        values = get_values(rewards, params.policy_params.discount).cuda()
        if len(advantages.shape) == 1: advantages = advantages[:, None]
        if len(values.shape) == 1: values = values[:, None]
        rolloutmem.append(old_states, new_states, raw_actions, rewards, dones, log_probs, advantages, values)
        episodes_rewards.append(episode_reward)
    return torch.mean(torch.Tensor(episodes_rewards))


# parallel_rollout = False


def parallel_rollout_env(rolloutmem, envs, actor, critic, params):
    # interact
    env_number = len(envs)
    env_attributes = ray.get(envs[0].get_attributes.remote())
    old_states, new_states, raw_actions, dones, rewards, log_probs, advantages, episode_reward \
        = [], [], [], [], [], [], [], [0] * env_number
    old_state = ray.get([env.reset.remote() for env in envs])
    rolloutmem.reset()
    for step in range(params.policy_params.horizon):
        # interact
        action, log_prob, raw_action = actor.gen_action(torch.Tensor(old_state).cuda())
        assert (env_attributes['action_high'].cuda() >= action).all() and (action >= env_attributes['action_low'].cuda()).all(), '>> Error: action value exceeds boundary!'
        # print("    >> action: {}".format(action))
        step_obs_batch = ray.get(
            [envs[i].step.remote(action[i].cpu()) for i in range(env_number)])  # new_obs, reward, done, info
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
        log_probs.append(log_prob.float())
        episode_reward = [float(reward[i]) + episode_reward[i] for i in range(len(reward))]
        # update old observation
        old_state = new_state
        if np.array(done).all():
            break
    dones[-1] = [True] * env_number
    if env_attributes['image_obs']:
        old_states = torch.Tensor(old_states).permute(1, 0, 2, 3, 4).cuda()
        new_states = torch.Tensor(new_states).permute(1, 0, 2, 3, 4).cuda()
        raw_actions = torch.stack(raw_actions).permute(1, 0, 2).detach().cuda()
        rewards = torch.Tensor(rewards).permute(1, 0).cuda()
        dones = torch.Tensor(dones).permute(1, 0).cuda()
        log_probs = torch.stack(log_probs).permute(1, 0, 2).detach().cuda()
    else:
        old_states = torch.Tensor(old_states).permute(1, 0, 2).cuda()
        new_states = torch.Tensor(new_states).permute(1, 0, 2).cuda()
        raw_actions = torch.stack(raw_actions).permute(1, 0, 2).detach().cuda()
        rewards = torch.Tensor(rewards).permute(1, 0).cuda()
        dones = torch.Tensor(dones).permute(1, 0).cuda()
        log_probs = torch.stack(log_probs).permute(1, 0, 2).detach().cuda()

    # gae_deltas = critic.gae_delta(old_states, new_states, rewards, .99).cuda()
    # advantages = get_advantage_new(gae_deltas, .99, .95).detach().cuda()[:, :, None]
    # values = get_values(rewards, .99).cuda()[:, :, None]
    # rolloutmem.append(old_states, new_states, raw_actions, rewards, dones, log_probs, advantages, values)
    for i in range(env_number):
        gae_deltas = critic.gae_delta(old_states[i], new_states[i], rewards[i], .99).cuda()
        advantages = get_advantage_new(gae_deltas, .99, .95)[:, None].detach().cuda()
        values = get_values(rewards[i], .99)[:, None].cuda()
        # abandon redundant step info
        first_done = (dones[i] > 0).nonzero().min()
        rolloutmem.append(old_states[i][:first_done + 1], new_states[i][:first_done + 1],
                          raw_actions[i][:first_done + 1], rewards[i][:first_done + 1], dones[i][:first_done + 1],
                          log_probs[i][:first_done + 1], advantages[i][:first_done + 1], values[i][:first_done + 1])
    return torch.mean(torch.Tensor(episode_reward))


def rollout(rolloutmem, envs, actor, critic, params):
    if params.parallel:
        # mean_episode_reward = rollout_parallel(rolloutmem, envs, actor, critic, params)
        mean_episode_reward = parallel_rollout_env(rolloutmem, envs, actor, critic, params)
    else:
        mean_episode_reward = rollout_serial(rolloutmem, envs, actor, critic, params)
    return mean_episode_reward


def optimize_step(optimizer, rolloutmem, actor, critic, params, iteration):
    entropy_discount = 1.
    if params.reducing_entro_loss:
        entropy_discount = 1. - iteration / params.iter_num
    # print('entropy_discount: {}'.format(entropy_discount))
    # sample rollout steps from current policy
    old_obs_batch, _, raw_action_batch, reward_batch, done_batch, old_log_prob_batch, advantage_batch, value_batch \
        = rolloutmem.sample(params.policy_params.batch_size)
    # compute loss factors
    logits = actor.policy_out(old_obs_batch)
    new_log_prob_batch = get_norm_log_prob(logits, raw_action_batch, actor.scale,
                                           dist_type=get_dist_type(params.env_name))
    assert len(new_log_prob_batch.shape) > 1, '    >>> [optimize_step -> new_log_prob_batch], Wrong Dimension'
    ratio = torch.exp(new_log_prob_batch - old_log_prob_batch)
    surr1 = ratio * advantage_batch
    surr2 = torch.clamp(ratio, 1 - params.policy_params.clip_param,
                        1 + params.policy_params.clip_param) * advantage_batch
    # compute losses
    policy_loss = - torch.mean(torch.min(surr1, surr2))
    critic_loss = torch.mean(torch.pow(critic.forward(old_obs_batch) - value_batch, 2))  # MSE loss
    entropy_loss = - torch.mean(get_entropy(logits, get_dist_type(params.env_name)))
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


def train(params, pretrains=None):
    # ============
    # Preparations
    # ============
    ray.init(log_to_driver=False, local_mode=False)  # or, ray.init()
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
        actor = gen_actor(params.env_name, params.policy_params.hidden_dim)
        critic = gen_critic(params.env_name, params.policy_params.hidden_dim)
        optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=0.0001)
        # load models
        print("\n\nLoading training checkpoint...")
        print("------------------------------")
        load_path = os.path.join(params.pretrain_file, params.prefix+'_iter_100.tar')
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
    for iteration in range(int(params.iter_num)):
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
        tb.add_scalar('loss', loss, iteration+iteration_pretrain)
        tb.add_scalar('policy_loss', -1 * policy_loss, iteration+iteration_pretrain)
        tb.add_scalar('critic_loss', critic_loss, iteration+iteration_pretrain)
        tb.add_scalar('entropy_loss', -1 * entropy_loss, iteration+iteration_pretrain)
        tb.add_scalar('advantage', advantage.mean(), iteration+iteration_pretrain)
        tb.add_scalar('ratio', ratio.mean(), iteration+iteration_pretrain)
        tb.add_scalar('surr1', surr1.mean(), iteration+iteration_pretrain)
        tb.add_scalar('surr2', surr2.mean(), iteration+iteration_pretrain)
        tb.add_scalar('epoch_len', epochs_len, iteration+iteration_pretrain)
        tb.add_scalar('reward', mean_iter_reward, iteration+iteration_pretrain)
        tb.add_scalar('reward_over_time(s)', mean_iter_reward, int(time.time() - time_start))
        iter_end_time = time.time()
        rollout_time.update(update_start_time - iter_start_time)
        update_time.update(iter_end_time - update_start_time)
        tb.add_scalar('rollout_time', rollout_time.val, iteration+iteration_pretrain)
        tb.add_scalar('update_time', update_time.val, iteration+iteration_pretrain)
        print('it {}: avgR: {:.3f} avgL: {:.3f} | rollout_time: {:.3f}sec update_time: {:.3f}sec'
              .format(iteration+iteration_pretrain, mean_iter_reward, epochs_len, rollout_time.val, update_time.val))
        # save rollout video
        if iteration % int(params.plotting_iters) == 0 and iteration > 0 and params.log_video:
            log_policy_rollout(params, actor, params.env_name, 'iter-{}'.format(iteration+iteration_pretrain))
        # save model
        if iteration % int(params.checkpoint_iter) == 0 and iteration > 0 and params.save_checkpoint:
            print("\n\nSaving training checkpoint...")
            print("-----------------------------")
            save_path = os.path.join("./save/model", params.prefix+'_iter_{}'.format(iteration+iteration_pretrain)+'.tar')
            torch.save({
                'iteration': iteration+iteration_pretrain,
                'seed': seed,
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'time_recorder': [rollout_time, update_time],
            }, save_path)
            print("Saved checkpoint to: {}".format(save_path))
            print("-----------------------------\n\n")

    # >> save rollout videos
    if params.log_video:
        for i in range(3):
            log_policy_rollout(params, actor, params.env_name, 'final-{}'.format(i))


