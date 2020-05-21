import torch, gym, time, os
from gym.wrappers import Monitor
from torch.distributions.normal import Normal
from networks import ActorContinueNet, CriticNet
from data import envnames_minigrid, envnames_classiccontrol, envnames_mujoco


def gen_env(env_name):
    env = gym.make(env_name)
    return env


def gen_actor(env_name, hidden_dim):
    # environment specific parameters
    env = gen_env(env_name)
    action_scale = env.action_space.high if hasattr(env.action_space, 'high') else 1
    # generate actor by environment name
    if env_name in envnames_mujoco:
        return ActorContinueNet(input_size=env.observation_space.shape[0],
                                output_size=env.action_space.shape[0],
                                hidden_dim=hidden_dim, action_scale=action_scale).cuda()
    else:
        raise NotImplementedError


def gen_critic(env_name, hidden_dim):
    # environment specific parameters
    env = gen_env(env_name)
    if env_name in envnames_mujoco:
        return CriticNet(input_size=env.observation_space.shape[0],
                         hidden_dim=hidden_dim).cuda()
    else:
        raise NotImplementedError


def count_model_params(model):
    return sum(p.numel() for p in model.parameters())


def get_advantage(horizon, gae_deltas, discount, lambd):
    step = len(gae_deltas) - 1
    advantage = 0.0
    while step >= horizon:
        advantage = lambd * discount * advantage
        advantage += gae_deltas[step]
        step -= 1
    return advantage.item()


def get_advantage_new(deltas, discount, lambd):
    advantage = [0 for i in range(len(deltas))]
    advantage[-1] = deltas[-1]
    for i in reversed(range(len(deltas) - 1)):
        advantage[i] = lambd * discount * advantage[i + 1] + deltas[i]
    return advantage


def get_values(rewards, discount):
    values = torch.zeros(len(rewards))
    for step in range(len(rewards)):
        index = len(rewards) - step - 1
        if step == 0:
            values[index] = rewards[index]
        else:
            values[index] = rewards[index] + discount * values[index + 1]
    return values


def get_entropy(logits):
    mean, cov = logits[0], logits[1]
    entropy = Normal(mean, cov).entropy()
    return entropy


def log_policy_rollout(params, actor, env, video_name):
    cur_time = time.strftime("[%Y-%m-%d_%H:%M:%S]", time.localtime())
    save_path_name = os.path.join(params.save_path, 'video', '{}->{}{}.mp4'
                                  .format(params.prefix, video_name, cur_time))
    env = Monitor(env, save_path_name, force=True)
    done = False
    episode_reward = 0.
    episode_length = 0.
    action_list = []
    observation = env.reset()
    print('Sampling trajectory...')
    while not done:
        action = actor.gen_action(torch.tensor(observation, dtype=torch.float32).cuda())[0]
        action_list.append(action)
        observation, reward, done, info = env.step(action.cpu())
        episode_reward += reward
        episode_length += 1
    print("Action Series: {}".format(action_list))
    print('Total reward:', episode_reward)
    print('Total length:', episode_length)
    env.close()
    print('Finished Sampling, saved video in {}.\n'.format(save_path_name))


class ParamDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count