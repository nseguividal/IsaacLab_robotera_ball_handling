# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Template: RSL-RL policy playback and evaluation script.

What goes here:
1. Parse CLI arguments (`--task`, `--num_envs`, `--checkpoint`, etc.) and Isaac Lab
   `AppLauncher` args, then launch `SimulationApp` BEFORE importing Isaac / Gym / RSL-RL.
2. Create the environment (`gym.make` -> `RslRlVecEnvWrapper`) and `OnPolicyRunner`.
3. Load the trained `.pt` checkpoint (`runner.load(checkpoint)`) and get the
   inference policy (`runner.get_inference_policy(device=...)`).
4. Step through `vec_env.step(actions)` inside `while simulation_app.is_running():`
   until stopped, then close the environment and `simulation_app`.
"""

import argparse
from isaaclab.app import AppLauncher


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser with playback and Isaac Lab launcher flags."""
    parser = argparse.ArgumentParser(description="Evaluate and visualize a trained RSL-RL policy.")
    parser.add_argument("--task", type=str, default="Isaac-Robotera-Ball-Rotation-v0", help="Gym task ID.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments to view.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt file).")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main():
    """Launch Isaac Sim and run the policy evaluation loop."""
    parser = build_arg_parser()
    args_cli = parser.parse_args()

    # 1. Launch SimulationApp before importing Isaac / Gym / RSL-RL modules
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # 2. Post-launch imports
    # import os
    # import torch
    # import gymnasium as gym
    # from rsl_rl.runners import OnPolicyRunner
    # from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    # import robotera_ball_handling.tasks
    # from robotera_ball_handling.tasks.ball_rotation.agents.rsl_rl_ppo_cfg import (
    #     RoboteraBallRotationPPORunnerCfg,
    # )

    # 3. TODO: Create environment & load checkpoint
    # env_cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    # env_cfg.scene.num_envs = args_cli.num_envs
    # env = gym.make(args_cli.task, cfg=env_cfg)
    # vec_env = RslRlVecEnvWrapper(env)
    #
    # agent_cfg = RoboteraBallRotationPPORunnerCfg()
    # runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # if args_cli.checkpoint and os.path.exists(args_cli.checkpoint):
    #     runner.load(args_cli.checkpoint)
    # policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 4. TODO: Rollout loop
    # obs, _ = vec_env.reset()
    # while simulation_app.is_running():
    #     with torch.inference_mode():
    #         actions = policy(obs)
    #         obs, rewards, dones, infos = vec_env.step(actions)

    # 5. Cleanup
    # env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
