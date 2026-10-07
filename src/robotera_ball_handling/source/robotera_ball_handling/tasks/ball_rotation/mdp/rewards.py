# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""MDP reward functions for in-hand dual-ball rotation."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def dual_ball_rotation_velocity(
    env: ManagerBasedRLEnv,
    target_velocity: float = 1.5,
    ball1_name: str = "ball1",
    ball2_name: str = "ball2",
) -> torch.Tensor:
    """Reward for rotating the two balls around each other at the target angular velocity.

    Computes planar angular velocity: omega_z = (r_x * v_y - r_y * v_x) / (r_x^2 + r_y^2)
    """
    ball1 = env.scene[ball1_name]
    ball2 = env.scene[ball2_name]

    # Relative position and velocity in XY plane
    r_xy = ball1.data.root_pos_w[:, :2] - ball2.data.root_pos_w[:, :2]
    v_xy = ball1.data.root_lin_vel_w[:, :2] - ball2.data.root_lin_vel_w[:, :2]

    r_sq = torch.sum(r_xy**2, dim=-1).clamp(min=1e-5)
    # Planar 2D cross product: r_x * v_y - r_y * v_x
    omega_z = (r_xy[:, 0] * v_xy[:, 1] - r_xy[:, 1] * v_xy[:, 0]) / r_sq

    # Gaussian tracking reward for target rotation velocity
    vel_error = torch.abs(omega_z - target_velocity)
    reward = torch.exp(-vel_error / 0.5)
    return reward


def dual_ball_containment(
    env: ManagerBasedRLEnv,
    palm_pos: tuple[float, float, float] = (0.05, 0.0, 0.42),
    ball1_name: str = "ball1",
    ball2_name: str = "ball2",
    std: float = 0.05,
) -> torch.Tensor:
    """Reward keeping the center of mass of the two balls centered in the palm."""
    ball1 = env.scene[ball1_name]
    ball2 = env.scene[ball2_name]

    com = (ball1.data.root_pos_w + ball2.data.root_pos_w) * 0.5
    target = torch.tensor(palm_pos, device=env.device).unsqueeze(0)
    dist = torch.norm(com - target, dim=-1)

    return torch.exp(-(dist**2) / (2 * std**2))


def dual_ball_proximity(
    env: ManagerBasedRLEnv,
    ball_diameter: float = 0.048,
    ball1_name: str = "ball1",
    ball2_name: str = "ball2",
    std: float = 0.02,
) -> torch.Tensor:
    """Reward keeping the two balls adjacent to each other without flying apart."""
    ball1 = env.scene[ball1_name]
    ball2 = env.scene[ball2_name]

    dist = torch.norm(ball1.data.root_pos_w - ball2.data.root_pos_w, dim=-1)
    dist_error = torch.abs(dist - ball_diameter)

    return torch.exp(-(dist_error**2) / (2 * std**2))

