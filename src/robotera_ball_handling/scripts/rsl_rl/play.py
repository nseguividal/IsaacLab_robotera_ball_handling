# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Script Purpose: Policy playback and evaluation script to load, run, and visualize trained RSL-RL checkpoints in the Isaac Lab environment.

Usage:
    python scripts/rsl_rl/play.py --checkpoint /path/to/model.pt
"""

import argparse
import os
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate and visualize a trained policy.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments to view.")
parser.add_argument("--task", type=str, default="Isaac-Robotera-Ball-Rotation-v0", help="Gym task ID.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt file).")

# Add Isaac Lab CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# Launch simulation
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Simulation imports after app launch"""
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit

# Import task module
import robotera_ball_handling.tasks
from robotera_ball_handling.tasks.ball_rotation.agents.rsl_rl_ppo_cfg import RoboteraBallRotationPPORunnerCfg


def main():
    """Run policy evaluation loop."""
    env_cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg)
    vec_env = RslRlVecEnvWrapper(env)

    agent_cfg = RoboteraBallRotationPPORunnerCfg()
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    # Load policy checkpoint
    if args_cli.checkpoint and os.path.exists(args_cli.checkpoint):
        print(f"[INFO] Loading checkpoint from: {args_cli.checkpoint}")
        runner.load(args_cli.checkpoint)
    else:
        print("[WARNING] No checkpoint provided; running with randomly initialized policy.")

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Reset environment
    obs, _ = vec_env.reset()

    print("\n[INFO] Running evaluation loop. Press Ctrl+C or close window to stop.\n")
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, infos = vec_env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

