import numpy as np
import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(
        self, state_dim, action_dim, hidden_num, hidden_layer, max_action, device
    ):
        super(PolicyNetwork, self).__init__()
        self.input_layer = nn.Linear(state_dim, hidden_layer)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_layer, hidden_layer) for _ in range(hidden_num)]
        )
        self.output_layer = nn.Linear(hidden_layer, action_dim)

        self.max_action = torch.tensor(max_action, dtype=torch.float, device=device)

    def forward(self, x):
        h = torch.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            h = torch.relu(hidden_layer(h))
        y = torch.tanh(self.output_layer(h))
        return y * self.max_action


class CriticNetwork(nn.Module):
    def __init__(
        self, state_dim, action_dim, epsilon_dim, hidden_num, hidden_layer, bias
    ):
        super(CriticNetwork, self).__init__()

        self.input_layer_1 = nn.Linear(
            state_dim + action_dim + epsilon_dim, hidden_layer
        )
        self.hidden_layers_1 = nn.ModuleList(
            [nn.Linear(hidden_layer, hidden_layer) for _ in range(hidden_num)]
        )
        self.output_layer_1 = nn.Linear(hidden_layer, 1)

        self.input_layer_2 = nn.Linear(
            state_dim + action_dim + epsilon_dim, hidden_layer
        )
        self.hidden_layers_2 = nn.ModuleList(
            [nn.Linear(hidden_layer, hidden_layer) for _ in range(hidden_num)]
        )
        self.output_layer_2 = nn.Linear(hidden_layer, 1)

        if bias != 0:
            print(self.output_layer_1.bias.data, self.output_layer_2.bias.data)
            self.output_layer_1.bias.data = torch.tensor(
                [bias], requires_grad=True, dtype=torch.float
            )
            self.output_layer_2.bias.data = torch.tensor(
                [bias], requires_grad=True, dtype=torch.float
            )
            print(self.output_layer_1.bias.data, self.output_layer_2.bias.data)

    def forward(self, state, action, epsilon):
        h1 = torch.relu(self.input_layer_1(torch.cat([state, action, epsilon], dim=1)))
        for hidden_layer in self.hidden_layers_1:
            h1 = torch.relu(hidden_layer(h1))
        y1 = self.output_layer_1(h1)

        h2 = torch.relu(self.input_layer_2(torch.cat([state, action, epsilon], dim=1)))
        for hidden_layer in self.hidden_layers_2:
            h2 = torch.relu(hidden_layer(h2))
        y2 = self.output_layer_2(h2)
        return y1, y2

    def Q1(self, state, action, epsilon):
        h1 = torch.relu(self.input_layer_1(torch.cat([state, action, epsilon], dim=1)))
        for hidden_layer in self.hidden_layers_1:
            h1 = torch.relu(hidden_layer(h1))
        y1 = self.output_layer_1(h1)
        return y1


class EtaNetwork(nn.Module):
    def __init__(
        self,
        omega_dim,
        min_omega,
        max_omega,
        hidden_num,
        hidden_layer,
        rand_state,
        device,
    ):
        super(EtaNetwork, self).__init__()
        self.hidden_num = hidden_num
        if hidden_num == 0:
            self.input_layer = nn.Linear(1, omega_dim, bias=False)
            initial_omega = rand_state.uniform(
                low=min_omega, high=max_omega, size=min_omega.shape
            )
            y2 = (initial_omega - min_omega) / np.maximum(
                max_omega - min_omega, np.ones(shape=min_omega.shape) * 0.00001
            )
            y1 = np.log(
                np.maximum(y2 / (1 - y2), np.ones(shape=min_omega.shape) * 0.00001)
            )
            for i in range(omega_dim):
                self.input_layer.weight.data[i] = y1[i]
        else:
            self.input_layer = nn.Linear(1, hidden_layer, bias=False)
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(hidden_layer, hidden_layer) for _ in range(hidden_num)]
            )
            self.output_layer = nn.Linear(hidden_layer, 1)
        self.min_epsilon = torch.tensor(min_omega, dtype=torch.float, device=device)
        self.max_epsilon = torch.tensor(max_omega, dtype=torch.float, device=device)

    def forward(self, x):
        y = self.input_layer(x)
        if self.hidden_num != 0:
            for hidden_layer in self.hidden_layers:
                y = torch.relu(hidden_layer(y))
            y = self.output_layer(y)
        y = torch.sigmoid(y)
        y = y * (self.max_epsilon - self.min_epsilon) + self.min_epsilon
        return y
