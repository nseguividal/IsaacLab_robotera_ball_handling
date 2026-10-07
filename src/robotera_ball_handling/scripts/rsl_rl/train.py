# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Training script for Robotera Ball Handling task with RSL-RL PPO."""

import argparse
import os
from datetime import datetime

# 1. App Launcher
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Robotera In-Hand Ball Rotation Policy.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel training environments.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--max_iterations", type=int, default=3000, help="Maximum training iterations.")
parser.add_argument("--task", type=str, default="Isaac-Robotera-Ball-Rotation-v0", help="Gym task ID.")

# Add Isaac Lab CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch simulation
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Simulation imports after app launch"""
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Import our registered tasks
import robotera_ball_handling.tasks
from robotera_ball_handling.tasks.ball_rotation.agents.rsl_rl_ppo_cfg import RoboteraBallRotationPPORunnerCfg


def main():
    """Train the agent."""
    # Create gymnasium environment
    env_cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    # Wrap in vector environment wrapper
    vec_env = RslRlVecEnvWrapper(env)

    # Agent hyperparameters
    agent_cfg = RoboteraBallRotationPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations

    # Log directory
    log_dir = os.path.join("logs", "rsl_rl", "robotera_ball_rotation", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    # Initialize RSL-RL PPO runner
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # Start training
    print(f"\n[INFO] Starting training with {args_cli.num_envs} environments on {agent_cfg.device}...")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

