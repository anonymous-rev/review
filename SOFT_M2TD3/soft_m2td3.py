import torch
import sys
import shutil
import os
import glob
import numpy as np
from omegaconf import OmegaConf

from SOFT_M2TD3.td3 import TD3
from SOFT_M2TD3.logger import Logger

sys.path.append("../")
from make_env import make_env
import gym
from collections import namedtuple

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done", "omega")
)


class SOFT_M2TD3:
    def __init__(self, config, experiment_name):
        self.config = config
        self.device = torch.device(
            config["system"]["device"] if torch.cuda.is_available() else "cpu"
        )
        self.experiment_name = experiment_name
        self.output_dir = "."
        cwd_dir = os.getcwd()
        if os.path.exists(f"{cwd_dir}/policies"):
            shutil.rmtree(f"{cwd_dir}/policies")
        if os.path.exists(f"{cwd_dir}/critics"):
            shutil.rmtree(f"{cwd_dir}/critics")
        file_list = glob.glob(f"{cwd_dir}/*")
        for file_path in file_list:
            if os.path.isfile(file_path):
                os.remove(file_path)

        os.makedirs(f"{self.output_dir}/policies", exist_ok=True)
        os.makedirs(f"{self.output_dir}/critics", exist_ok=True)
        OmegaConf.save(config, f"{self.output_dir}/hyperparameter.yaml")

        env = make_env(
            self.config["environment"]["env_name"],
            [],
            np.array(self.config["environment"]["change_param_min"]),
            self.config["system"]["seed"],
            self.config["algorithm"]["name"],
            self.config["environment"]["xml_file"],
            self.config["xml_name"],
        )

        if isinstance(env.observation_space, gym.spaces.Dict):
            self.state_dim = len(env.observation_space.spaces)
        else:
            self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = env.action_space.high[0]
        self.step = 0
        self.episode_len = env._max_episode_steps

        torch.manual_seed(config["system"]["seed"])
        self.rand_state = np.random.RandomState(config["system"]["seed"])

        self.agent = TD3(
            config,
            self.state_dim,
            self.action_dim,
            len(self.config["environment"]["change_param_min"]),
            self.max_action,
            self.rand_state,
            self.device,
        )
        self.logger = Logger()
        self.change_param_min = np.array(self.config["environment"]["change_param_min"])
        self.change_param_max = np.array(self.config["environment"]["change_param_max"])

    def save_model(self):
        torch.save(
            self.agent.policyNet.to("cpu").state_dict(),
            f"{self.output_dir}/policies/policy-{self.step}.pkl",
        )
        self.agent.policyNet.to(self.device)
        torch.save(
            self.agent.critic_p.to("cpu").state_dict(),
            f"{self.output_dir}/critics/critic-{self.step}-{self.experiment_name}.pkl",
        )
        self.agent.critic_p.to(self.device)

    def sample_omega(self, step):
        if step <= self.config["algorithm"]["start_steps"]:
            omega = self.rand_state.uniform(
                low=self.change_param_min,
                high=self.change_param_max,
                size=len(self.change_param_min),
            )
            dis_restart_flag = "None"
            prob_restart_flag = "None"
        else:
            omega, dis_restart_flag, prob_restart_flag = self.agent.get_omega()
            if dis_restart_flag:
                dis_restart_flag = "True"
            else:
                dis_restart_flag = "False"
            if prob_restart_flag:
                prob_restart_flag = "True"
            else:
                prob_restart_flag = "False"

        assert (
            len(omega)
            == len(self.config["environment"]["change_param_min"])
            == len(self.config["environment"]["change_param_max"])
        )

        omega = np.clip(
            omega,
            self.config["environment"]["change_param_min"],
            self.config["environment"]["change_param_max"],
        )
        assert isinstance(omega, np.ndarray)

        return omega, dis_restart_flag, prob_restart_flag

    def interact(self, env, omega):
        state = env.reset()
        total_reward = 0
        for _ in range(env._max_episode_steps):
            self.logger.update_step(self.agent)
            if self.step <= self.config["algorithm"]["start_steps"]:
                action = self.rand_state.uniform(
                    low=env.action_space.low,
                    high=env.action_space.high,
                    size=env.action_space.low.shape,
                ).astype(env.action_space.low.dtype)
            else:
                action = self.agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            self.agent.add_memory(state, action, next_state, reward, done, omega)
            if self.agent.omega_std >= self.agent.min_omega_std:
                self.agent.omega_std = list(
                    np.array(self.agent.omega_std) - self.agent.omega_std_step
                )
            if self.step >= self.config["algorithm"]["start_steps"]:
                self.agent.train(self.step)
            state = next_state
            total_reward += reward

            if self.step % self.config["evaluation"]["evaluate_qvalue_interval"] == 0:
                self.logger.update_qvalue(self.agent)
            if self.step % self.config["evaluation"]["logger_interval"] == 0:
                self.logger.dump(self.output_dir)
            if self.step % self.config["evaluation"]["evaluate_interval"] == 0:
                self.save_model()
            if self.step >= self.config["algorithm"]["max_steps"]:
                return True, None
            self.step += 1
            self.episode_len += 1
            if done:
                return False, total_reward
        return False, total_reward

    def main(self):
        while True:
            self.agent.current_episode_len = self.episode_len
            self.episode_len = 0
            omega, dis_restart_flag, prob_restart_flag = self.sample_omega(self.step)
            env = make_env(
                self.config["environment"]["env_name"],
                self.config["environment"]["change_param_names"],
                omega,
                self.config["system"]["seed"],
                self.config["algorithm"]["name"],
                self.config["environment"]["xml_file"],
                self.config["xml_name"],
            )
            _ = env.seed(self.config["system"]["seed"])
            flag, total_reward = self.interact(env, omega)
            print(self.step, omega, total_reward)
            self.logger.update_episode(dis_restart_flag, prob_restart_flag)
            if flag:
                break

        self.logger.dump(self.output_dir)
        return self.logger


if __name__ == "__main__":
    assert False
