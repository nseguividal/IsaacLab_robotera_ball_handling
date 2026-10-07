# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Ball rotation task registration for Robotera Hand."""

import gymnasium as gym

from .ball_rotation_env_cfg import RoboteraBallRotationEnvCfg

# Register environment into Gymnasium
gym.register(
    id="Isaac-Robotera-Ball-Rotation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RoboteraBallRotationEnvCfg,
    },
)

__all__ = ["RoboteraBallRotationEnvCfg"]

