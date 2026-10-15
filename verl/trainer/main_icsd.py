"""Entry point for Influence Calibration for Self-Distillation."""

import hydra

from verl.trainer.main_privileged_sd import run_privileged_sd


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_privileged_sd(config, "icsd")


if __name__ == "__main__":
    main()
