from torch.nn import functional as F
from torch import nn
import torch

from math import sqrt


def hidden_layer_init(layer):
    fan_in = layer.weight.data.size(0)
    limit = 1 / sqrt(fan_in)
    return -limit, limit


class Actor(nn.Module):
    def __init__(self, observation_size, action_size, fc1_units=256, out_init_range=3e-3, seed=0):
        super().__init__()

        # Random seed
        torch.manual_seed(seed)

        # Layers
        self.fc1 = nn.Linear(observation_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)

        # Initialize
        self.reset_parameters(out_init_range)

    def reset_parameters(self, out_init_range):
        # Initialize weights and biases
        self.fc1.weight.data.uniform_(*hidden_layer_init(self.fc1))
        # self.fc1.bias.data.uniform_(*hidden_layer_init(self.fc1))

        # Initialize output layer weights and biases
        self.fc2.weight.data.uniform_(-out_init_range, out_init_range)
        # self.fc2.bias.data.uniform_(-out_init_range, out_init_range)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return F.tanh(self.fc2(x))


class Critic(nn.Module):
    def __init__(self, observation_size, action_size, fc1_units=256, fc2_units=256, fc3_units=128,
                 out_init_range=3e-3, seed=0):
        super().__init__()

        # Random seed
        torch.manual_seed(seed)

        # Layers
        self.fc1 = nn.Linear(observation_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)

        # Initialize
        self.reset_parameters(out_init_range)

    def reset_parameters(self, out_init_range):
        # Initialize weights and biases
        self.fc1.weight.data.uniform_(*hidden_layer_init(self.fc1))
        # self.fc1.bias.data.uniform_(*hidden_layer_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_layer_init(self.fc2))
        # self.fc2.bias.data.uniform_(*hidden_layer_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_layer_init(self.fc3))
        # self.fc3.bias.data.uniform_(*hidden_layer_init(self.fc3))

        # Initialize output layer weights and biases
        self.fc4.weight.data.uniform_(-out_init_range, out_init_range)
        # self.fc4.bias.data.uniform_(-out_init_range, out_init_range)

    def forward(self, state, actions):
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(torch.cat((x, actions), dim=1)))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)
