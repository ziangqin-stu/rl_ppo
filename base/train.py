from utils import gen_env, gen_actor, gen_critic, count_model_params, get_advantage, get_values, get_entropy, \
    log_policy_rollout, AverageMeter
from rolloutmemory import RolloutMemory
from networks import get_norm_log_prob

import torch
from torch.utils.tensorboard import SummaryWriter
import time
import multiprocessing as mp
import threading


def rollout(rolloutmem, envs, actor, critic, params):
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
            torch.Tensor(old_states).cuda(), torch.Tensor(new_states).cuda(), torch.Tensor(raw_actions).cuda(), \
            torch.Tensor(rewards).cuda(), torch.Tensor(dones).cuda(), torch.Tensor(log_probs).cuda()
        # compute loss factors
        gae_deltas = critic.gae_delta(old_states, new_states, rewards, params.policy_params.discount)
        for t in range(len(gae_deltas)):
            advantages.append(get_advantage(t, gae_deltas, params.policy_params.discount, params.policy_params.lambd))
        advantages = torch.Tensor(advantages)
        # advantages = torch.Tensor([get_advantage(step, gae_deltas, params.policy_params.discount, params.policy_params.lambd)
        #               for step in range(len(gae_deltas))])
        values = get_values(rewards, params.policy_params.discount)
        # store epoch
        rolloutmem.append(old_states, new_states, raw_actions, rewards, dones, log_probs, advantages, values)
        # record epoch reward
        episodes_rewards.append(episode_reward)

    return torch.mean(torch.Tensor(episodes_rewards))


def optimize_step(optimizer, rolloutmem, actor, critic, param):
    # sample rollout steps from current policy
    old_obs_batch, _, raw_action_batch, reward_batch, done_batch, old_log_prob_batch, advantage_batch, value_batch \
        = rolloutmem.sample(param.policy_params.batch_size)
    # compute loss factors
    mean, cov = actor.policy_out(old_obs_batch)
    mean, cov = mean.view(-1), cov.view(-1)
    new_log_prob_batch = get_norm_log_prob([mean, cov], raw_action_batch, actor.scale).view(-1)
    ratio = torch.exp(new_log_prob_batch - old_log_prob_batch)
    surr1 = ratio * advantage_batch
    surr2 = torch.clamp(ratio, 1 - param.policy_params.clip_param, 1 + param.policy_params.clip_param) * advantage_batch
    # compute losses
    policy_loss = - torch.mean(torch.min(surr1, surr2))
    critic_loss = torch.mean(torch.pow(critic.forward(old_obs_batch).view(-1) - value_batch, 2))  # MSE loss
    entropy_loss = - torch.mean(get_entropy([mean, cov]))
    loss = policy_loss + param.policy_params.critic_coef * critic_loss + param.policy_params.entropy_coef * entropy_loss
    # gradient descent
    optimizer.zero_grad()
    loss.backward()
    for param in list(actor.parameters()) + list(critic.parameters()):
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    # return variables for logger
    return loss, policy_loss, critic_loss, entropy_loss, advantage_batch, ratio, surr1, surr2, \
           torch.mean(rolloutmem.compute_epoch_length())


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
        # mean_iter_reward = rollout(rolloutmem, envs, actor, critic, params)
        mean_iter_reward = rollout(rolloutmem, envs, actor, critic, params)
        # optimize by gradient descent
        update_start_time = time.time()
        for epoch in range(params.policy_params.epochs_num):
            loss, policy_loss, critic_loss, entropy_loss, advantage, ratio, surr1, surr2, epochs_len = \
                optimize_step(optimizer, rolloutmem, actor, critic, params)
            tb.add_scalar('loss', loss, iteration * params.policy_params.epochs_num + epoch + 1)
            tb.add_scalar('policy_loss', -1 * policy_loss, iteration * params.policy_params.epochs_num + epoch + 1)
            tb.add_scalar('critic_loss', critic_loss, iteration * params.policy_params.epochs_num + epoch + 1)
            tb.add_scalar('entropy_loss', -1 * entropy_loss, iteration * params.policy_params.epochs_num + epoch + 1)
            tb.add_scalar('advantage', advantage.mean(), iteration * params.policy_params.epochs_num + epoch + 1)
            tb.add_scalar('ratio', ratio.mean(), iteration * params.policy_params.epochs_num + epoch + 1)
            tb.add_scalar('surr1', surr1.mean(), iteration * params.policy_params.epochs_num + epoch + 1)
            tb.add_scalar('surr2', surr2.mean(), iteration * params.policy_params.epochs_num + epoch + 1)
            tb.add_scalar('epoch_len', epochs_len, iteration * params.policy_params.epochs_num + epoch + 1)
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
