from torch import nn
import torch

from math import sqrt


def fan_in_init(tensor):
    limit = 1 / sqrt(tensor.size(-1))
    nn.init.uniform_(tensor, -limit, limit)


class Actor(nn.Module):
    def __init__(self, observation_space, action_space, fc1_units, fc2_units, fc3_units,
                 out_init_range=1):
        super().__init__()

        # Layers
        self.fc1 = nn.Linear(observation_space, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_space)

        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # Initialize weights and biases
        fan_in_init(self.fc1.weight)
        fan_in_init(self.fc1.bias)
        fan_in_init(self.fc2.weight)
        fan_in_init(self.fc2.bias)
        fan_in_init(self.fc3.weight)
        fan_in_init(self.fc3.bias)

        # Initialize output layer weights and biases
        nn.init.uniform_(self.fc4.weight, -out_init_range, out_init_range)
        nn.init.uniform_(self.fc4.bias, -out_init_range, out_init_range)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.tanh(self.fc4(x))


class Critic(nn.Module):
    def __init__(self, observation_space, action_space, fc1_units, fc2_units, fc3_units,
                 out_init_range=1):
        super().__init__()

        # Layers
        self.fc1 = nn.Linear(observation_space, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_space, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)

        # Activation functions
        self.relu = nn.ReLU()

        # Initialize weights and biases
        fan_in_init(self.fc1.weight)
        fan_in_init(self.fc1.bias)
        fan_in_init(self.fc2.weight)
        fan_in_init(self.fc2.bias)
        fan_in_init(self.fc3.weight)
        fan_in_init(self.fc3.bias)

        # Initialize output layer weights and biases
        nn.init.uniform_(self.fc4.weight, -out_init_range, out_init_range)
        nn.init.uniform_(self.fc4.bias, -out_init_range, out_init_range)

    def forward(self, state, actions):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(torch.cat([x, actions], 1)))
        x = self.relu(self.fc3(x))
        return self.fc4(x)
