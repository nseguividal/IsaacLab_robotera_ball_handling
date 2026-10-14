# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Visuo-tactile optical sensor configurations modeled after Touchlab & GelSight sensors."""

import torch
import torch.nn.functional as F
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg


# ==============================================================================
# Optical Visuo-Tactile Sensor Array (5 Fingertips)
# ==============================================================================
# Modeled after Touchlab / GelSight optical tactile sensors:
# - Placed on the contact pad of each fingertip looking outward towards the manipulation space
# - Micro-optical sensing on all 5 fingertips (Thumb, Index, Middle, Ring, Pinky)
# - Macro depth clipping range: 1mm to 25mm (0.001m to 0.025m) for contact surface deformation
# - 32x32 resolution per fingertip (high tactile detail, GPU-tensor optimized for RL)
# ==============================================================================

VISUO_TACTILE_FINGERTIPS_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/right_hand_.*_tip/visuo_tactile",
    offset=TiledCameraCfg.OffsetCfg(
        # Positioned flush on the outer fingertip pad surface facing outward along +Y, Up pointing towards distal tip (-Z)
        pos=(0.0, 0.0087, 0.008),
        rot=(0.0, 0.7071, 0.7071, 0.0),  # Forward: +Y (outward), Up: -Z (distal tip), Left: +X (thumb)
        convention="world",
    ),
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.5,
        horizontal_aperture=3.0,
        vertical_aperture=3.0,
        clipping_range=(0.0001, 0.015),  # 0.1mm to 15mm continuous tracking range
    ),
    width=32,
    height=32,
)


# ==============================================================================
# Fingertip Contact Force & Pressure Reporter (Physics Sensor)
# ==============================================================================
# Complements the visuo-tactile camera by providing direct 3D contact force vectors
# (Fx, Fy, Fz) and normal force magnitudes for reward calculation and physics analysis.
# ==============================================================================

FINGERTIP_CONTACT_SENSOR_CFG = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/right_hand_.*_link2",
    history_length=3,
    update_period=0.0,
    track_air_time=False,
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball1", "{ENV_REGEX_NS}/Ball2"],
)


# ==============================================================================
# Visuo-Tactile to Spatial Force Array Converter
# ==============================================================================
def compute_visuo_tactile_force_array(
    tactile_depth: torch.Tensor,
    contact_threshold: float = 0.003,  # 3mm contact zone threshold (last 3mm as elastomer gel)
    gel_stiffness: float = 80.0,       # Realistic N/m elasticity scaling
    output_taxels: tuple[int, int] = (8, 8),
    contact_forces: torch.Tensor | None = None,
) -> torch.Tensor:
    """Converts optical visuo-tactile depth maps into a calibrated spatial 2D force array per fingertip.

    Args:
        tactile_depth: Raw depth tensor from visuo-tactile cameras (num_cameras, H, W, 1),
                       (num_cameras, H, W), or (num_envs, 5, H, W).
        contact_threshold: Distance threshold below which contact indentation is registered.
        gel_stiffness: Effective elastic stiffness scaling (N/m).
        output_taxels: Spatial taxel grid resolution (e.g., 8x8 matrix per fingertip).
        contact_forces: Optional direct physics normal forces (num_envs, 5) or (num_cameras,) for calibration.

    Returns:
        torch.Tensor: Force array in Newtons per taxel of shape (*batch_dims, taxel_h, taxel_w).
    """
    # 0. Squeeze trailing channel dimension if present (e.g. [N, H, W, 1] -> [N, H, W])
    if tactile_depth.ndim >= 3 and tactile_depth.shape[-1] == 1:
        tactile_depth = tactile_depth.squeeze(-1)

    # 1. Clean non-finite values (NaN / Inf become contact_threshold -> 0.0 indentation)
    cleaned_depth = torch.nan_to_num(tactile_depth, nan=contact_threshold, posinf=contact_threshold, neginf=0.0)

    # 2. Surface indentation: Maximum when depth is ~0 (deep contact), 0.0 when depth >= contact_threshold
    valid_contact_mask = (cleaned_depth >= 0.0) & (cleaned_depth < contact_threshold)
    raw_indentation = torch.where(
        valid_contact_mask,
        (contact_threshold - cleaned_depth.clamp(min=0.0)),
        torch.zeros_like(cleaned_depth),
    )

    # 3. Downsample to taxel array grid (e.g. 8x8 force matrix)
    batch_shape = raw_indentation.shape[:-2]
    h_orig, w_orig = raw_indentation.shape[-2], raw_indentation.shape[-1]

    if (h_orig, w_orig) != output_taxels:
        flat_indent = raw_indentation.reshape(-1, 1, h_orig, w_orig)
        pooled = F.adaptive_avg_pool2d(flat_indent, output_taxels)
        indentation = pooled.view(*batch_shape, *output_taxels)
    else:
        indentation = raw_indentation

    # 4. Compute normal force per taxel (Newtons) from optical gel deformation
    force_array = indentation * gel_stiffness

    # 5. If ground-truth physics contact force is supplied, gate and calibrate magnitude
    if contact_forces is not None:
        if contact_forces.ndim >= 1 and contact_forces.shape[-1] == 3:
            contact_forces = torch.norm(contact_forces, dim=-1)
        # Reshape contact forces to align with batch dimensions (*batch_shape, 1, 1)
        c_forces = contact_forces.reshape(*batch_shape, 1, 1)

        in_contact = (c_forces > 0.01)
        taxel_max = torch.amax(force_array, dim=(-2, -1), keepdim=True)
        
        # When physical contact sensor reports force, scale tactile distribution so peak square = peak contact force
        scaled_forces = torch.where(
            taxel_max > 1e-4,
            force_array * (c_forces / taxel_max.clamp(min=1e-4)),
            torch.full_like(force_array, 1.0) * c_forces,
        )
        # Strictly zero out forces if physics contact is 0 (open air, no contact with target object)
        force_array = torch.where(in_contact, scaled_forces, torch.zeros_like(force_array))

    return force_array


def compute_visuo_tactile_contact_features(
    tactile_depth: torch.Tensor,
    contact_threshold: float = 0.003,
    gel_stiffness: float = 80.0,
    output_taxels: tuple[int, int] = (8, 8),
    contact_forces: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts both full force array and aggregate fingertip metrics (total force per finger).

    Returns:
        force_array: (num_envs, 5, taxel_h, taxel_w)
        total_forces: (num_envs, 5) total normal peak force in Newtons per finger
    """
    force_array = compute_visuo_tactile_force_array(
        tactile_depth,
        contact_threshold=contact_threshold,
        gel_stiffness=gel_stiffness,
        output_taxels=output_taxels,
        contact_forces=contact_forces,
    )
    # Maximum over spatial taxel dimensions (H, W) to get the total peak contact force felt by the finger
    total_forces = torch.amax(force_array, dim=(-2, -1))
    return force_array, total_forces


