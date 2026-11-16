# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Task registration for the ball rotation environment.

What goes here:
- Import your environment config class from `ball_rotation_env_cfg.py`.
- Register the task ID with `gymnasium.register(...)` so scripts (e.g., `train.py`,
  `play.py`, or `gym.make(...)`) can discover and instantiate the environment.
"""

import gymnasium as gym

from .ball_rotation_env_cfg import RoboteraBallRotationEnvCfg

gym.register(
    id="Isaac-Robotera-Ball-Rotation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RoboteraBallRotationEnvCfg,
    },
)

__all__ = ["RoboteraBallRotationEnvCfg"]
