import pandas as pd
import os


class Logger:
    def __init__(self):
        self.reset()
        self.step = 0
        self.episode = 0

    def reset(self):
        self.step_list = []
        self.episode_list = []
        self.update_omega_list = []
        self.qvalue_step_list = []
        self.qvalue_list = []
        self.dis_restarts = []
        self.prob_restarts = []

    def update_step(self, agent):
        self.step_list.append(self.step)
        self.update_omega_list.append(agent.update_omega)
        self.step += 1

    def update_qvalue(self, agent):
        self.qvalue_step_list.append(self.step)
        q_value_list = agent.get_qvalue_list()
        self.qvalue_list.append(q_value_list)

    def update_episode(self, dis_restart_flag, prob_restart_flag):
        self.episode_list.append(self.episode)
        self.dis_restarts.append(dis_restart_flag)
        self.prob_restarts.append(prob_restart_flag)
        self.episode += 1

    def dump(self, output_dir):
        dic = {
            "step": self.step_list,
            "update_omega": self.update_omega_list,
        }
        df = pd.DataFrame(dic)
        if os.path.exists(f"{output_dir}/step_log.csv"):
            df.to_csv(f"{output_dir}/step_log.csv", mode="a", header=False, index=False)
        else:
            df.to_csv(f"{output_dir}/step_log.csv", mode="w", index=False)

        dic = {
            "qvalue_step": self.qvalue_step_list,
            "qvalue_list": self.qvalue_list,
        }
        df = pd.DataFrame(dic)
        if os.path.exists(f"{output_dir}/qvalue_log.csv"):
            df.to_csv(
                f"{output_dir}/qvalue_log.csv", mode="a", header=False, index=False
            )
        else:
            df.to_csv(f"{output_dir}/qvalue_log.csv", mode="w", index=False)

        dic = {
            "episode": self.episode_list,
            "dis_restarts": self.dis_restarts,
            "prob_restarts": self.prob_restarts,
        }
        df = pd.DataFrame(dic)
        if os.path.exists(f"{output_dir}/episode_log.csv"):
            df.to_csv(
                f"{output_dir}/episode_log.csv", mode="a", header=False, index=False
            )
        else:
            df.to_csv(f"{output_dir}/episode_log.csv", mode="w", index=False)

        self.reset()
