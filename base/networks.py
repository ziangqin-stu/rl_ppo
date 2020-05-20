import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


class FCContinueBasicNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, action_scale):
        super(FCContinueBasicNet, self).__init__()
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
        self.scale = torch.tensor([action_scale]).float().cuda()
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
        # if torch.isnan(std[0]):
        #     print(std)
        return mean, cov


class FCBasicNet(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(FCBasicNet, self).__init__()
        # fully connected network
        self.fc1 = nn.Linear(input_size, 2 * hidden_dim)
        self.fc2 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.fc3 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.fc4 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)
        # initialize network parameters
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.orthogonal_(self.fc4.weight)
        nn.init.orthogonal_(self.fc5.weight)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# class ActorContinueNet(FCContinueBasicNet):
#     def __init__(self, input_size, output_size, hidden_dim, action_scale):
#         super(ActorContinueNet, self).__init__(input_size, output_size, hidden_dim, action_scale)
#
#     def forward(self, state):
#         return super().forward(state)
#
#     def gen_action(self, state):
#         mean, std = self.forward(state)
#         dist = Normal(mean, std)
#         raw_action = dist.sample()
#         action = self.scale * torch.tanh(raw_action)
#         log_prob = get_norm_log_prob([mean, std], raw_action, self.scale).view(-1)
#         return action, log_prob, raw_action
class ActorContinueNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, action_scale):
        super(ActorContinueNet, self).__init__()
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
        self.scale = torch.tensor([action_scale]).float().cuda()
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


# class CriticNet(FCBasicNet):
#     def __init__(self, input_size, hidden_dim):
#         super(CriticNet, self).__init__(input_size, hidden_dim)
#
#     def forward(self, state):
#         return super().forward(state)
#
#     def gae_delta(self, old_state, new_state, rewards, discount):
#         return rewards + discount * self.forward(new_state).view(-1) - self.forward(old_state).view(-1)
class CriticNet(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(CriticNet, self).__init__()
        # fully connected network
        self.fc1 = nn.Linear(input_size, 2 * hidden_dim)
        self.fc2 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.fc3 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.fc4 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)
        # initialize network parameters
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.orthogonal_(self.fc4.weight)
        nn.init.orthogonal_(self.fc5.weight)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def gae_delta(self, old_state, new_state, rewards, discount):
        return rewards + discount * self.forward(new_state).view(-1) - self.forward(old_state).view(-1)



def get_norm_log_prob(logits, raw_actions, scale):
    mean, cov = logits[0], logits[1]
    action_batch = scale * torch.tanh(raw_actions)
    # log_prob = torch.log(1 / scale) - 2 * torch.log(1 - (action_batch / scale) ** 2 + 1e-6).view(-1) \
    #        + Normal(mean, cov).log_prob(raw_actions.view(-1))
    log_prob = torch.log(1 / scale) - 2 * torch.log(1 - (action_batch / scale) ** 2 + 1e-6) \
               + Normal(mean, cov).log_prob(raw_actions)
    return log_prob

