from collections import namedtuple
import copy
import itertools

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from M2TD3.replay_buffer import ReplayBuffer
from M2TD3.network import (
    PolicyNetwork,
    CriticNetwork,
    EtaNetwork,
)

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done", "omega")
)


class TD3:
    def __init__(
        self, config, state_dim, action_dim, omega_dim, max_action, rand_state, device
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.omega_dim = omega_dim
        self.min_omega = np.array(config["environment"]["change_param_min"])
        self.max_omega = np.array(config["environment"]["change_param_max"])
        self.min_omega_tensor = torch.tensor(
            config["environment"]["change_param_min"], dtype=torch.float, device=device
        )
        self.max_omega_tensor = torch.tensor(
            config["environment"]["change_param_max"], dtype=torch.float, device=device
        )
        self.max_action = max_action

        self.config = config

        self.device = device
        self.rand_state = rand_state

        self.policy_std = config["algorithm"]["policy_std_rate"] * max_action
        self.policy_noise = config["algorithm"]["policy_noise_rate"] * max_action
        self.noise_clip_policy = (
            config["algorithm"]["noise_clip_policy_rate"] * max_action
        )

        self.eta_input = torch.tensor([[1]], dtype=torch.float, device=self.device)
        self.eta_input_batch = torch.tensor(
            [[1] * self.config["algorithm"]["batch_size"]],
            dtype=torch.float,
            device=self.device,
        ).view(self.config["algorithm"]["batch_size"], 1)

        self.omega_std = list(
            config["algorithm"]["omega_std_rate"]
            * (self.max_omega - self.min_omega)
            / 2
        )
        self.min_omega_std = list(
            config["algorithm"]["min_omega_std_rate"]
            * (self.max_omega - self.min_omega)
            / 2
        )
        self.omega_std_step = (
            np.array(self.omega_std) - np.array(self.min_omega_std)
        ) / (self.config["algorithm"]["max_steps"])

        self.omega_noise = (
            config["algorithm"]["policy_noise_rate"]
            * (self.max_omega - self.min_omega)
            / 2
        )
        self.noise_clip_omega = torch.tensor(
            config["algorithm"]["noise_clip_omega_rate"]
            * (self.max_omega - self.min_omega)
            / 2,
            device=self.device,
            dtype=torch.float,
        )

        self.replay_buffer = ReplayBuffer(
            rand_state, capacity=config["algorithm"]["replay_size"]
        )

        self.etaNet_prob = [
            1 / config["network"]["eta_parameter_num"]
            for _ in range(config["network"]["eta_parameter_num"])
        ]
        self.element_list = [i for i in range(config["network"]["eta_parameter_num"])]

        self._init_network(
            state_dim, action_dim, config["environment"]["dim"], max_action, config
        )
        self._init_optimizer(config)
        self.update_omega = [0 for _ in range(len(self.max_omega))]

    def _init_network(self, state_dim, action_dim, omega_dim, max_action, config):
        self.policyNet = PolicyNetwork(
            state_dim,
            action_dim,
            config["network"]["policy_hidden_num"],
            config["network"]["policy_hidden_size"],
            max_action,
            self.device,
        ).to(self.device)

        self.critic_p = CriticNetwork(
            state_dim,
            action_dim,
            omega_dim,
            config["network"]["critic_hidden_num"],
            config["network"]["critic_hidden_size"],
            config["network"]["p_bias"],
        ).to(self.device)

        self.etaNet_list = [None for _ in range(config["network"]["eta_parameter_num"])]
        self.optimizer_envNet_list = [
            None for _ in range(config["network"]["eta_parameter_num"])
        ]
        for i in range(config["network"]["eta_parameter_num"]):
            self._init_etaNet(i)

        self.policy_target = copy.deepcopy(self.policyNet)
        self.critic_target_p = copy.deepcopy(self.critic_p)

    def _init_optimizer(self, config):
        self.optimizer_policy = optim.Adam(
            self.policyNet.parameters(), lr=config["algorithm"]["p_lr"]
        )
        self.optimizer_critic_p = optim.Adam(
            self.critic_p.parameters(), lr=config["algorithm"]["q_lr"]
        )

    def add_memory(self, *args):
        transition = Transition(*args)
        self.replay_buffer.append(transition)

    def get_action(self, state, greedy=False):
        state_tensor = torch.tensor(state, dtype=torch.float, device=self.device).view(
            -1, self.state_dim
        )
        action = self.policyNet(state_tensor)
        if not greedy:
            noise = torch.tensor(
                self.rand_state.normal(0, self.policy_std),
                dtype=torch.float,
                device=self.device,
            )
            action = action + noise
        return action.squeeze(0).detach().cpu().numpy()

    def get_omega(self, greedy=False):
        dis_restart_flag = False
        prob_restart_flag = False
        if self.config["algorithm"]["restart_distance"]:
            change_env_parameter_index_list = self._calc_diff()
            for index in change_env_parameter_index_list:
                self._init_etaNet(index)
                self._init_etaNet_prob(index)
                dis_restart_flag = True
        if self.config["algorithm"]["restart_probability"]:
            change_env_parameter_index_list = self._minimum_prob()
            for index in change_env_parameter_index_list:
                self._init_etaNet(index)
                self._init_etaNet_prob(index)
                prob_restart_flag = True

        etaNet_index = self._select_etaNet()
        env_parameter = self.etaNet_list[etaNet_index](self.eta_input)

        if not greedy:
            noise = torch.tensor(
                self.rand_state.normal(0, self.omega_std),
                dtype=torch.float,
                device=self.device,
            )
            env_parameter += noise
        return (
            env_parameter.squeeze(0).detach().cpu().numpy(),
            dis_restart_flag,
            prob_restart_flag,
        )

    def _buffer2batch(self):
        transitions = self.replay_buffer.sample(self.config["algorithm"]["batch_size"])
        if transitions is None:
            return None, None, None, None, None, None
        batch = Transition(*zip(*transitions))
        state_batch = torch.tensor(batch.state, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(batch.action, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(
            batch.next_state, device=self.device, dtype=torch.float
        )
        reward_batch = torch.tensor(
            batch.reward, device=self.device, dtype=torch.float
        ).unsqueeze(1)
        not_done = np.array([(not don) for don in batch.done])
        not_done_batch = torch.tensor(
            not_done, device=self.device, dtype=torch.float
        ).unsqueeze(1)
        omega_batch = torch.tensor(batch.omega, device=self.device, dtype=torch.float)
        return (
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            not_done_batch,
            omega_batch,
        )

    def train(self, step):
        (
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            not_done_batch,
            omega_batch,
        ) = self._buffer2batch()
        if state_batch is None:
            return None, None, None

        self._update_critic(
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            not_done_batch,
            omega_batch,
        )
        if step % self.config["algorithm"]["policy_freq"] == 0:
            self._update_actor(state_batch)

            self._update_target()

    def _update_critic(
        self,
        state_batch,
        action_batch,
        next_state_batch,
        reward_batch,
        not_done_batch,
        omega_batch,
    ):
        with torch.no_grad():
            action_noise = (torch.randn_like(action_batch) * self.policy_noise).clamp(
                -self.noise_clip_policy, self.noise_clip_policy
            )
            next_action_batch = (
                self.policy_target(next_state_batch) + action_noise
            ).clamp(-self.max_action, self.max_action)
            omega_noise = torch.max(
                torch.min(
                    (
                        torch.randn_like(omega_batch)
                        * torch.tensor(
                            self.omega_noise, device=self.device, dtype=torch.float
                        )
                    ),
                    self.noise_clip_omega,
                ),
                -self.noise_clip_omega,
            )
            next_omega_batch = torch.max(
                torch.min((omega_batch + omega_noise), self.max_omega_tensor),
                self.min_omega_tensor,
            )

            targetQ1, targetQ2 = self.critic_target_p(
                next_state_batch, next_action_batch, next_omega_batch
            )
            targetQ = torch.min(targetQ1, targetQ2)
            targetQ = (
                reward_batch
                + not_done_batch * self.config["algorithm"]["gamma"] * targetQ
            )

        currentQ1, currentQ2 = self.critic_p(state_batch, action_batch, omega_batch)
        critic_loss = F.mse_loss(currentQ1, targetQ) + F.mse_loss(currentQ2, targetQ)

        self.optimizer_critic_p.zero_grad()
        critic_loss.backward()
        self.optimizer_critic_p.step()

    def _update_actor(self, state_batch):

        worst_policy_loss_index = None
        worst_policy_loss = None
        worst_policy_loss_value = -np.inf
        for eta_index in range(self.config["network"]["eta_parameter_num"]):
            envParam_batch = self.etaNet_list[eta_index](self.eta_input_batch)

            policy_loss = -self.critic_p.Q1(
                state_batch, self.policyNet(state_batch), envParam_batch.detach()
            ).mean()
            if policy_loss.item() >= worst_policy_loss_value:
                self.update_omega = list(
                    self.etaNet_list[eta_index](self.eta_input)
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )
                worst_policy_loss = policy_loss
                worst_policy_loss_index = eta_index
                worst_policy_loss_value = policy_loss.item()

        envParam_batch = self.etaNet_list[worst_policy_loss_index](self.eta_input_batch)

        env_loss = self.critic_p.Q1(
            state_batch, self.policyNet(state_batch).detach(), envParam_batch
        ).mean()
        self.optimizer_envNet_list[worst_policy_loss_index].zero_grad()
        env_loss.backward()
        self.optimizer_envNet_list[worst_policy_loss_index].step()

        self.optimizer_policy.zero_grad()
        worst_policy_loss.backward()
        self.optimizer_policy.step()

        self._update_env_prob(worst_policy_loss_index)

    def _update_target(self):
        for target_param, param in zip(
            self.critic_target_p.parameters(), self.critic_p.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config["algorithm"]["polyak"])
                + param.data * self.config["algorithm"]["polyak"]
            )

        for target_param, param in zip(
            self.policy_target.parameters(), self.policyNet.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config["algorithm"]["polyak"])
                + param.data * self.config["algorithm"]["polyak"]
            )

    def _calc_diff(self):
        change_env_parameter_index_list = []
        eta_parameter_list = []
        for i in range(self.config["network"]["eta_parameter_num"]):
            eta_parameter_list.append(
                self.etaNet_list[i](self.eta_input)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
        for eta_pair in itertools.combinations(eta_parameter_list, 2):
            distance = np.linalg.norm(
                np.array(eta_pair[0]) - np.array(eta_pair[1]), ord=1
            )
            if distance <= self.config["algorithm"]["eta_parameter_distance"]:
                change_env_parameter_index_list.append(
                    eta_parameter_list.index(eta_pair[0])
                )
        return change_env_parameter_index_list

    def _update_env_prob(self, index):
        p = [0] * self.config["network"]["eta_parameter_num"]
        p[index] = 1
        coeff = 1 / self.current_episode_len
        for i in range(self.config["network"]["eta_parameter_num"]):
            self.etaNet_prob[i] = self.etaNet_prob[i] * (1 - coeff) + coeff * p[i]

    def _minimum_prob(self):
        indexes = []
        for index in range(self.config["network"]["eta_parameter_num"]):
            prob = self.etaNet_prob[index]
            if prob < self.config["algorithm"]["minimum_prob"]:
                indexes.append(index)
        return indexes

    def _init_etaNet(self, index):
        envNet = EtaNetwork(
            self.omega_dim,
            self.min_omega,
            self.max_omega,
            self.config["network"]["eta_hidden_num"],
            self.config["network"]["eta_hidden_size"],
            self.rand_state,
            self.device,
        ).to(self.device)
        optimizer_env = optim.Adam(
            envNet.parameters(), lr=self.config["algorithm"]["e_lr"]
        )
        self.etaNet_list[index] = envNet
        self.optimizer_envNet_list[index] = optimizer_env

    def _init_etaNet_prob(self, index):
        self.etaNet_prob[index] = 0
        sum_prob = sum(self.etaNet_prob)
        p = sum_prob / (self.config["network"]["eta_parameter_num"] - 1)
        self.etaNet_prob[index] = p

    def _select_etaNet(self):
        self.etaNet_prob = list(np.array(self.etaNet_prob) / sum(self.etaNet_prob))
        select_index = self.rand_state.choice(
            a=self.element_list, size=1, p=self.etaNet_prob
        )
        return select_index[0]

    def get_qvalue_list(self):
        qvalue_list = []
        transitions = self.replay_buffer.sample(self.config["algorithm"]["batch_size"])
        for eta_index in range(self.config["network"]["eta_parameter_num"]):
            if transitions is None:
                qvalue_list.append(0)
                continue
            batch = Transition(*zip(*transitions))
            state_batch = torch.tensor(
                batch.state, device=self.device, dtype=torch.float
            )
            q_value = self.critic_p.Q1(
                state_batch,
                self.policyNet(state_batch),
                self.etaNet_list[eta_index](self.eta_input_batch),
            ).mean()
            qvalue_list.append(q_value.item())
        return qvalue_list
