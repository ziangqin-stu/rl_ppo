import math

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


class ActorContinueFC(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, action_scale):
        super(ActorContinueFC, self).__init__()
        # mean
        self.mean_fc1 = nn.Linear(input_size, hidden_dim)
        self.mean_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_fc8 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_fc9 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_fc10 = nn.Linear(hidden_dim, output_size)
        # covariance
        self.cov_fc1 = nn.Linear(input_size, hidden_dim // 2)
        self.cov_fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.cov_fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.cov_fc4 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.cov_fc5 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.cov_fc6 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.cov_fc7 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.cov_fc8 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.cov_fc9 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.cov_fc10 = nn.Linear(hidden_dim // 2, output_size)
        # action scale
        self.scale = torch.tensor([action_scale]).float().cuda()
        # initialize network parameters
        nn.init.orthogonal_(self.mean_fc1.weight)
        nn.init.orthogonal_(self.mean_fc2.weight)
        nn.init.orthogonal_(self.mean_fc3.weight)
        nn.init.orthogonal_(self.mean_fc4.weight)
        nn.init.orthogonal_(self.mean_fc5.weight)
        nn.init.orthogonal_(self.mean_fc6.weight)
        nn.init.orthogonal_(self.mean_fc7.weight)
        nn.init.orthogonal_(self.mean_fc8.weight)
        nn.init.orthogonal_(self.mean_fc9.weight)
        nn.init.orthogonal_(self.mean_fc10.weight)
        nn.init.orthogonal_(self.cov_fc1.weight)
        nn.init.orthogonal_(self.cov_fc2.weight)
        nn.init.orthogonal_(self.cov_fc3.weight)
        nn.init.orthogonal_(self.cov_fc4.weight)
        nn.init.orthogonal_(self.cov_fc5.weight)
        nn.init.orthogonal_(self.cov_fc6.weight)
        nn.init.orthogonal_(self.cov_fc7.weight)
        nn.init.orthogonal_(self.cov_fc8.weight)
        nn.init.orthogonal_(self.cov_fc9.weight)
        nn.init.orthogonal_(self.cov_fc10.weight)

    def forward(self, state):
        mean = torch.relu(self.mean_fc1(state))
        mean = torch.relu(self.mean_fc2(mean))
        mean = torch.relu(self.mean_fc3(mean))
        mean = torch.relu(self.mean_fc4(mean))
        mean = torch.relu(self.mean_fc5(mean))
        mean = torch.relu(self.mean_fc6(mean))
        mean = torch.relu(self.mean_fc7(mean))
        mean = torch.relu(self.mean_fc8(mean))
        mean = torch.relu(self.mean_fc9(mean))
        mean = self.mean_fc10(mean)
        cov = torch.relu(self.cov_fc1(state))
        cov = torch.relu(self.cov_fc2(cov))
        cov = torch.relu(self.cov_fc3(cov))
        cov = torch.relu(self.cov_fc4(cov))
        cov = torch.relu(self.cov_fc5(cov))
        cov = torch.relu(self.cov_fc6(cov))
        cov = torch.relu(self.cov_fc7(cov))
        cov = torch.relu(self.cov_fc8(cov))
        cov = torch.relu(self.cov_fc9(cov))
        cov = torch.exp(self.cov_fc10(cov))
        # cov = torch.clamp(cov, 1e-6, 1.0e30)
        return mean, cov

    def gen_action(self, state):
        mean, cov = self.forward(state)
        dist = Normal(mean.double(), cov.double())
        raw_action = dist.sample()
        action = self.scale * torch.tanh(raw_action)
        log_prob = get_norm_log_prob([mean, cov], raw_action, self.scale, dist_type='Normal')
        return action.cpu(), log_prob, raw_action

    def policy_out(self, state):
        mean, cov = self.forward(state)
        # cov = torch.clamp(cov, 1e-6, 1.0e30)
        return mean.double(), cov.double()


class CriticFC(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(CriticFC, self).__init__()
        # fully connected network
        self.fc1 = nn.Linear(input_size, 2 * hidden_dim)
        self.fc2 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.fc3 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.fc4 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.fc5 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.fc6 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.fc7 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.fc8 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.fc9 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc10 = nn.Linear(hidden_dim, 1)
        # initialize network parameters
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.orthogonal_(self.fc4.weight)
        nn.init.orthogonal_(self.fc5.weight)
        nn.init.orthogonal_(self.fc6.weight)
        nn.init.orthogonal_(self.fc7.weight)
        nn.init.orthogonal_(self.fc8.weight)
        nn.init.orthogonal_(self.fc9.weight)
        nn.init.orthogonal_(self.fc10.weight)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = torch.relu(self.fc9(x))
        x = self.fc10(x)
        return x

    def gae_delta(self, old_state, new_state, rewards, discount):
        # return rewards + discount * self.forward(new_state).view(-1) - self.forward(old_state).view(-1)
        return rewards + discount * self.forward(new_state).squeeze(-1) - self.forward(old_state).squeeze(-1)


class ActorDiscreteFC(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, action_scale):
        # def fc layers
        super(ActorDiscreteFC, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_size)
        self.scale = torch.tensor([action_scale]).cuda()
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

    def gen_action(self, state):
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        raw_action = dist.sample()
        action = int(self.scale * raw_action)
        log_prob = get_norm_log_prob(logits, raw_action, self.scale, dist_type='Categorical')
        return action, log_prob, raw_action

    def policy_out(self, state):
        logits = self.forward(state)
        return logits


class CNNDiscreteNet(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(CNNDiscreteNet, self).__init__()
        w, h, r = input_shape[1], input_shape[0], input_shape[2]
        # build CNN layers
        self.conv1 = nn.Sequential(  # (r, w, h)
            nn.Conv2d(
                in_channels=r,
                out_channels=8,
                kernel_size=5,
                padding=2,  # padding=(kernel_size-1)/2 when stride=1
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=7,
                padding=3,
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=5,
                padding=2,
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        conv_dim = lambda x, f, p, s: math.floor((x - f + 2 * p) / s) + 1
        wo = conv_dim(conv_dim(conv_dim(conv_dim(conv_dim(conv_dim(w, 5, 2, 1), 2, 0, 2), 7, 3, 1), 2, 0, 2), 5, 2, 1),
                      2, 0, 2)
        ho = conv_dim(conv_dim(conv_dim(conv_dim(conv_dim(conv_dim(h, 5, 2, 1), 2, 0, 2), 7, 3, 1), 2, 0, 2), 5, 2, 1),
                      2, 0, 2)
        conv_out_dim = int(16 * wo * ho)
        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_out_dim, out_features=conv_out_dim // 5), nn.ReLU(),
            nn.Linear(in_features=conv_out_dim // 5, out_features=conv_out_dim // 50), nn.ReLU(),
            nn.Linear(in_features=conv_out_dim // 50, out_features=60), nn.ReLU(),
            nn.Linear(in_features=60, out_features=output_dim), nn.ReLU()
        )
        # initialize network parameters
        for i in range(len(self.conv1)):
            if hasattr(self.conv1[i], 'weight'):
                nn.init.orthogonal_(self.conv1[i].weight)
        for i in range(len(self.conv2)):
            if hasattr(self.conv2[i], 'weight'):
                nn.init.orthogonal_(self.conv2[i].weight)
        for i in range(len(self.conv3)):
            if hasattr(self.conv3[i], 'weight'):
                nn.init.orthogonal_(self.conv3[i].weight)
        for i in range(len(self.fc)):
            if hasattr(self.fc[i], 'weight'):
                nn.init.orthogonal_(self.fc[i].weight)

    def forward(self, state):
        if len(state.shape) == 3:
            # reformat single image: (width, height, channel) -> (channel, width, height)
            state = state.permute(2, 0, 1)
            state = state.unsqueeze(0)
        elif len(state.shape) == 4:
            # reformat batch images: (batch_size, width, height, channel) -> (batch_size, channel, width, height)
            state = state.permute(0, 3, 1, 2)
        elif len(state.shape) == 5:
            # reformat batch images: (batch_size, parallel_batch_size, width, height, channel) -> (batch_size, parallel_batch_size, channel, width, height)
            state = state.permute(0, 1, 4, 2, 3)
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


class ActorDiscreteCNN(CNNDiscreteNet):
    def __init__(self, input_shape, output_size, action_scale):
        super(ActorDiscreteCNN, self).__init__(input_shape, output_size)
        self.scale = torch.tensor([action_scale]).float().cuda()

    def gen_action(self, state):
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        raw_action = dist.sample()
        action = self.scale * raw_action
        log_prob = dist.log_prob(raw_action.double())
        if len(action.shape) < 2: action = action[:, None]  # expand dimension of minigrid action(int)
        if len(raw_action.shape) < 2: raw_action = raw_action[:, None]  # also raw_action
        if len(log_prob.shape) < 2: log_prob = log_prob[:, None]  # also log_prob
        return action, log_prob, raw_action

    def policy_out(self, state):
        x = self.forward(state)
        return x


class CriticCNN(CNNDiscreteNet):
    def __init__(self, input_shape):
        super(CriticCNN, self).__init__(input_shape, 1)

    def forward(self, state):
        x = super().forward(state)
        return x

    def gae_delta(self, old_state, new_state, rewards, discount):
        return rewards + discount * self.forward(new_state).view(-1) - self.forward(old_state).view(-1)


def get_norm_log_prob(logits, raw_actions, scale, dist_type):
    log_prob = None
    if dist_type is 'Normal':
        mean, cov = logits[0], logits[1]
        # cov value may overflow
        if not torch.stack([~torch.isnan(cov[i]) for i in range(len(cov))]).bool().all():
            print("    >>> nan in cov, clipped ratio value.")
            cov = torch.clamp(cov, 0., 3.0e38)
        action_batch = scale * torch.tanh(raw_actions)
        assert (-1. <= torch.stack([action_batch / scale])).all() and (torch.stack([action_batch / scale <= 1.])).all(), \
            "    >>> [get_norm_log_prob], action value error."
        log_prob = torch.log(1 / scale) - 2 * torch.log(1 - (action_batch / scale) ** 2 + 1e-6) \
                   + Normal(mean.double(), cov.double()).log_prob(raw_actions)
        # log_prob = torch.prod(log_prob, dim=1)[:, None]
    elif dist_type is 'Categorical':
        # Categorical.log_prob accepts tight-dimension data, return 1-dimension tensor when received batch input,
        #   use "view(-1)" to unbox input data
        log_prob = Categorical(logits=logits.double()).log_prob(raw_actions.view(-1))
        # add redundant dimension for standardization in project when return batch data
        if log_prob.shape[0] > 1:
            log_prob = log_prob[:, None]
    return log_prob
