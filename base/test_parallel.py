from rolloutmemory import RolloutMemory
from utils import gen_actor, gen_critic, gen_env, get_advantage, get_values
import time, torch


def rollout_serial(env_name, env_number, horizon, seed):
    # prepare rollout tools
    envs = [gen_env(env_name) for i in range(env_number)]
    for i in range(env_number): envs[i].seed(seed=seed)
    actor = gen_actor(env_name, 32)
    critic = gen_critic(env_name, 32)
    rolloutmem = RolloutMemory(env_number * horizon, env_name)
    # rollout
    time_start = time.time()
    episodes_rewards = []
    for env in envs:
        time_1 = time.time()
        old_obs = env.reset()
        states_1, states_2, raw_actions, dones, rewards, log_probs, advantages, episode_reward \
            = [], [], [], [], [], [], [], 0.
        actor_time, interact_time, collect_time, loop_time= 0., 0., 0., 0.
        time_e = time.time()
        for step in range(horizon):
            # act one step in current environment
            time_a = time.time()
            action, log_prob, raw_action = actor.gen_action(torch.Tensor(old_obs).cuda())
            time_b = time.time()
            new_obs, reward, done, info = env.step(action.cpu())
            time.sleep(.002)  # check this issue: https://github.com/openai/mujoco-py/issues/340
            # record trajectory step
            time_c = time.time()
            states_1.append(old_obs)
            states_2.append(new_obs)
            raw_actions.append(raw_action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            episode_reward += reward
            # update old observation
            old_obs = new_obs
            # if done:
            # break
            time_d = time.time()
            actor_time += time_b - time_a
            interact_time += time_c - time_b
            collect_time += time_d - time_c
            loop_time += time_a - time_e
            time_e = time.time()
        dones[-1] = True
        # reformat trajectory step
        time_2 = time.time()
        states_1, states_2, raw_actions, rewards, dones, log_probs = \
            torch.Tensor(states_1).cuda(), torch.Tensor(states_2).cuda(), torch.Tensor(raw_actions).cuda(), \
            torch.Tensor(rewards).cuda(), torch.Tensor(dones).cuda(), torch.Tensor(log_probs).cuda()
        # compute loss factors
        gae_deltas = critic.gae_delta(states_1, states_2, rewards, 0.99)
        for t in range(len(gae_deltas)):
            advantages.append(get_advantage(t, gae_deltas, 0.99, 0.95))
        advantages = torch.Tensor(advantages)
        values = get_values(rewards, 0.99)
        # store epoch
        rolloutmem.append(states_1, states_2, raw_actions, rewards, dones, log_probs, advantages, values)
        # record epoch reward
        episodes_rewards.append(episode_reward)
        time_3 = time.time()
        print("avgR: {:.3f}, rollout_time: {:.3f}, compute_time: {:.3f}, actor_time: {:.3f}, interact_time: {:.3f}, collect_time: {:.3f}, loop_time: {:.3f}"
              .format(torch.Tensor(episodes_rewards).mean(), time_2 - time_1, time_3 - time_2,actor_time, interact_time, collect_time, loop_time))
    time_end = time.time()
    print("total_rollout_time: {:.3f}".format(time_end - time_start))


def rollout_parallel(env_name, env_number, horizon, seed):
    # prepare rollout tools
    envs = [gen_env(env_name) for i in range(env_number)]
    for i in range(env_number): envs[i].seed(seed=seed)
    actor = gen_actor(env_name, 32)
    critic = gen_critic(env_name, 32)
    rolloutmem = RolloutMemory(env_number * horizon, env_name)
    # rollout
    time_start = time.time()
    episodes_rewards = []
    print("developing...")


rollout_serial('InvertedPendulum-v2', 1, 1000, 123)
