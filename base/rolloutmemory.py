import torch
import numpy as np
from utils import gen_env


class RolloutMemory:
    """
    Test:
        - check data dimension
        - check data type
        - check data device
        - check data value
    """

    def __init__(self, capacity, env_name):
        # environment specific parameters
        self.env = gen_env(env_name)
        action = self.env.action_space.sample()
        act_dim = action.shape if isinstance(action, np.ndarray) else (1,)
        obs_dim = self.env.observation_space.shape
        # memories
        self.env_name = env_name
        self.offset = 0
        self.capacity = int(capacity)
        self.old_obs_mem = torch.zeros(capacity, *obs_dim).cuda() if obs_dim[0] > 1 else torch.zeros(capacity).cuda()
        self.new_obs_mem = torch.zeros(capacity, *obs_dim).cuda() if obs_dim[0] > 1 else torch.zeros(capacity).cuda()
        self.action_mem = torch.zeros(capacity, *act_dim).cuda()
        self.reward_mem = torch.zeros(capacity).cuda()
        self.done_mem = torch.zeros(capacity).cuda()
        self.log_prob_mem = torch.zeros(capacity, *act_dim).cuda()
        self.advantage_mem = torch.zeros(capacity, 1).cuda()
        self.value_mem = torch.zeros(capacity, 1).cuda()
        self.epochs_len = torch.zeros(capacity, 1).cuda()

    def reset(self):
        self.__init__(self.capacity, self.env_name)

    def append(self, old_obs_batch, new_obs_batch, action_batch, reward_batch, done_batch,
               log_prob_batch, advantage_batch, value_batch):
        # insert segment boundary
        batch_size = len(old_obs_batch)
        start, end = self.offset, self.offset + batch_size
        # insert batches to memories
        self.old_obs_mem[start: end] = old_obs_batch[:]
        self.new_obs_mem[start: end] = new_obs_batch[:]
        self.action_mem[start: end] = action_batch[:]
        self.reward_mem[start: end] = reward_batch[:]
        self.done_mem[start: end] = done_batch[:]
        self.log_prob_mem[start: end] = log_prob_batch[:]
        self.advantage_mem[start: end] = advantage_batch[:]
        self.value_mem[start: end] = value_batch[:]
        self.offset += batch_size

    def sample(self, batch_size):
        out = np.random.choice(self.offset, batch_size, replace=False)
        return (
            self.old_obs_mem[out],
            self.new_obs_mem[out],
            self.action_mem[out],
            self.reward_mem[out],
            self.done_mem[out],
            self.log_prob_mem[out],
            self.advantage_mem[out],
            self.value_mem[out],
        )

    def compute_epoch_length(self):
        done_indexes = (self.done_mem == 1).nonzero()
        self.epochs_len = torch.Tensor([done_indexes[i] - done_indexes[i - 1] if i > 0 else done_indexes[0]
                                        for i in reversed(range(len(done_indexes)))])
        return self.epochs_len
