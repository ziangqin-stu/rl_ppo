import ray
import torch
from rolloutmemory import RolloutMemory
from utils import gen_env, gen_actor, gen_critic, ParallelEnv, AverageMeter
from torch.utils.tensorboard import SummaryWriter
import os


def test_state_dict(env_name):
    env = gen_env(env_name)
    actor = gen_actor(env_name, 64)
    critic = gen_critic(env_name, 64)
    rolloutmem = RolloutMemory(50 * 200, env_name)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=0.0001)
    tb = SummaryWriter()

    for param_tensor in actor.state_dict():
        print(param_tensor, "\t", actor.state_dict()[param_tensor].size())
    for param_tensor in critic.state_dict():
        print(param_tensor, "\t", critic.state_dict()[param_tensor].size())
    for param_tensor in optimizer.state_dict():
        print(param_tensor, "\t", optimizer.state_dict()[param_tensor])


def test_save(env_name):
    iteration = 1000
    actor = gen_actor(env_name, 64)
    critic = gen_critic(env_name, 64)
    rolloutmem = RolloutMemory(5 * 10, env_name)
    envs = [ParallelEnv.remote(env_name, i) for i in range(5)]
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()),
                                 lr=0.0001)
    seed = 123
    tb = SummaryWriter()
    for i in range(100): tb.add_scalar('loss', i, i)
    rollout_time, update_time = AverageMeter(), AverageMeter()
    rollout_time.update(100)
    update_time.update(100)

    save_path = os.path.join("../base/save/model", 'dev_Hopper_resume.tar')
    torch.save({
        'iteration': iteration,
        'seed': seed,
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rolloutmem': rolloutmem,
        'time_recorder': [rollout_time, update_time],
    }, save_path)
    print("Save Done!")


def test_load(env_name):
    actor = gen_actor(env_name, 64)
    critic = gen_critic(env_name, 64)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=0.0001)

    load_path = os.path.join("../base/save/model", 'dev_Hopper_resume.tar')
    checkpoint = torch.load(load_path)

    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.train()
    critic.load_state_dict(checkpoint['critic_state_dict'])
    critic.train()
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    rolloutmem = checkpoint['rolloutmem']
    iteration = checkpoint['iteration']
    seed = checkpoint['seed']
    [rollout_time, update_time] = checkpoint['time_recorder']
    print("Load Done!")
    print('')



ray.init()
# test_state_dict("Hopper-v2")
test_save("Hopper-v2")
test_load("Hopper-v2")
# test_state_dict("Hopper-v2")
