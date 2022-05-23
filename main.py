import hydra

from M2TD3.m2td3 import M2TD3
from SOFT_M2TD3.soft_m2td3 import SOFT_M2TD3

ALGORITHM_DICT = {
    "M2TD3": M2TD3,
    "SOFT_M2TD3": SOFT_M2TD3,
}


@hydra.main(config_path="configs", config_name="default")
def main(config):
    experiment_name = config["experiment_name"]
    algorithm = ALGORITHM_DICT[config["algorithm"]["name"]](config, experiment_name)
    algorithm.main()


if __name__ == "__main__":
    main()
