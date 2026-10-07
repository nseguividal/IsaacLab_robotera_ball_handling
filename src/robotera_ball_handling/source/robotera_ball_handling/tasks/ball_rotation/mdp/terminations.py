# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""MDP termination functions for in-hand dual-ball rotation."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_dropped(
    env: ManagerBasedRLEnv,
    min_height: float = 0.32,
    max_dist_from_palm: float = 0.12,
    palm_pos: tuple[float, float, float] = (0.05, 0.0, 0.42),
    ball1_name: str = "ball1",
    ball2_name: str = "ball2",
) -> torch.Tensor:
    """Terminate episode if either ball falls off the hand or moves too far from palm."""
    ball1 = env.scene[ball1_name]
    ball2 = env.scene[ball2_name]

    target = torch.tensor(palm_pos, device=env.device).unsqueeze(0)

    # Check height (fall below palm level)
    b1_dropped_z = ball1.data.root_pos_w[:, 2] < min_height
    b2_dropped_z = ball2.data.root_pos_w[:, 2] < min_height

    # Check radius distance from palm center
    b1_dist = torch.norm(ball1.data.root_pos_w[:, :2] - target[:, :2], dim=-1) > max_dist_from_palm
    b2_dist = torch.norm(ball2.data.root_pos_w[:, :2] - target[:, :2], dim=-1) > max_dist_from_palm

    return b1_dropped_z | b2_dropped_z | b1_dist | b2_dist

