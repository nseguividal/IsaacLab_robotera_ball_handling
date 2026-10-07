# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""MDP observation functions for in-hand dual-ball rotation."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from robotera_ball_handling.sensors import compute_visuo_tactile_contact_features, compute_visuo_tactile_force_array

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def dual_ball_state(
    env: ManagerBasedRLEnv,
    palm_pos: tuple[float, float, float] = (0.05, 0.0, 0.42),
    ball1_name: str = "ball1",
    ball2_name: str = "ball2",
) -> torch.Tensor:
    """Relative 6D pose and velocity of both balls relative to palm."""
    ball1 = env.scene[ball1_name]
    ball2 = env.scene[ball2_name]

    palm = torch.tensor(palm_pos, device=env.device).unsqueeze(0)

    # Relative positions
    b1_rel_pos = ball1.data.root_pos_w - palm
    b2_rel_pos = ball2.data.root_pos_w - palm

    # Quaternions
    b1_quat = ball1.data.root_quat_w
    b2_quat = ball2.data.root_quat_w

    # Velocities
    b1_lin_vel = ball1.data.root_lin_vel_w
    b2_lin_vel = ball2.data.root_lin_vel_w
    b1_ang_vel = ball1.data.root_ang_vel_w
    b2_ang_vel = ball2.data.root_ang_vel_w

    return torch.cat(
        [
            b1_rel_pos,
            b1_quat,
            b1_lin_vel,
            b1_ang_vel,
            b2_rel_pos,
            b2_quat,
            b2_lin_vel,
            b2_ang_vel,
        ],
        dim=-1,
    )


def visuo_tactile_force_array_obs(
    env: ManagerBasedRLEnv,
    sensor_name: str = "tactile_cameras",
    taxels: tuple[int, int] = (8, 8),
    contact_sensor_name: str | None = "contact_forces",
) -> torch.Tensor:
    """Flattened 2D force array across all 5 fingertips extracted from visuo-tactile depth."""
    tactile_sensor = env.scene[sensor_name]
    tactile_depth = tactile_sensor.data.output["distance_to_image_plane"]

    contact_forces = None
    if contact_sensor_name and contact_sensor_name in env.scene.sensors:
        contact_forces = torch.norm(env.scene[contact_sensor_name].data.net_forces_w, dim=-1)

    # Group cameras by environment: (num_envs * 5, ...) -> (num_envs, 5, ...)
    if tactile_depth.shape[0] == env.num_envs * 5:
        if tactile_depth.ndim == 4 and tactile_depth.shape[-1] == 1:
            tactile_depth = tactile_depth.squeeze(-1)
        tactile_depth = tactile_depth.view(env.num_envs, 5, *tactile_depth.shape[-2:])

    # Compute (num_envs, 5, 8, 8) force array
    force_array = compute_visuo_tactile_force_array(
        tactile_depth, output_taxels=taxels, contact_forces=contact_forces
    )
    # Flatten spatial taxels for policy network: (num_envs, 5 * 8 * 8 = 320)
    return force_array.view(env.num_envs, -1)


def visuo_tactile_total_forces_obs(
    env: ManagerBasedRLEnv,
    sensor_name: str = "tactile_cameras",
    contact_sensor_name: str | None = "contact_forces",
) -> torch.Tensor:
    """Total normal force (Newtons) on each of the 5 fingertips (num_envs, 5)."""
    tactile_sensor = env.scene[sensor_name]
    tactile_depth = tactile_sensor.data.output["distance_to_image_plane"]

    contact_forces = None
    if contact_sensor_name and contact_sensor_name in env.scene.sensors:
        contact_forces = torch.norm(env.scene[contact_sensor_name].data.net_forces_w, dim=-1)

    if tactile_depth.shape[0] == env.num_envs * 5:
        if tactile_depth.ndim == 4 and tactile_depth.shape[-1] == 1:
            tactile_depth = tactile_depth.squeeze(-1)
        tactile_depth = tactile_depth.view(env.num_envs, 5, *tactile_depth.shape[-2:])

    _, total_forces = compute_visuo_tactile_contact_features(
        tactile_depth, contact_forces=contact_forces
    )
    return total_forces.view(env.num_envs, 5)
