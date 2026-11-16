# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Template: RSL-RL PPO training script for Isaac Lab tasks.

What goes here:
1. Parse CLI arguments (`--task`, `--num_envs`, `--seed`, `--max_iterations`, etc.)
   and Isaac Lab `AppLauncher` args, then launch `SimulationApp` BEFORE importing
   Isaac Lab / Gym / RSL-RL modules.
2. Load the registered environment config (`gym.spec(task).kwargs["env_cfg_entry_point"]`)
   and create the vectorized environment (`gym.make` -> `RslRlVecEnvWrapper`).
3. Instantiate your runner config (e.g. `RoboteraBallRotationPPORunnerCfg`) and
   `OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=..., device=...)`.
4. Call `runner.learn(...)` and close the environment and `simulation_app`.
"""

import argparse
from isaaclab.app import AppLauncher


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser with training and Isaac Lab launcher flags."""
    parser = argparse.ArgumentParser(description="Train an RL policy with RSL-RL.")
    parser.add_argument("--task", type=str, default="Isaac-Robotera-Ball-Rotation-v0", help="Gym task ID.")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_iterations", type=int, default=3000, help="Maximum training iterations.")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main():
    """Launch Isaac Sim and run the RSL-RL training loop."""
    parser = build_arg_parser()
    args_cli = parser.parse_args()

    # 1. Launch SimulationApp before importing Isaac / Gym / RSL-RL modules
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # 2. Post-launch imports
    # import os
    # from datetime import datetime
    # import gymnasium as gym
    # from rsl_rl.runners import OnPolicyRunner
    # from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    # import robotera_ball_handling.tasks
    # from robotera_ball_handling.tasks.ball_rotation.agents.rsl_rl_ppo_cfg import (
    #     RoboteraBallRotationPPORunnerCfg,
    # )

    # 3. TODO: Create environment & wrap with RslRlVecEnvWrapper
    # env_cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    # env_cfg.scene.num_envs = args_cli.num_envs
    # env_cfg.seed = args_cli.seed
    # env = gym.make(args_cli.task, cfg=env_cfg)
    # vec_env = RslRlVecEnvWrapper(env)

    # 4. TODO: Initialize OnPolicyRunner and start training
    # agent_cfg = RoboteraBallRotationPPORunnerCfg()
    # agent_cfg.max_iterations = args_cli.max_iterations
    # log_dir = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # 5. Cleanup
    # env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
