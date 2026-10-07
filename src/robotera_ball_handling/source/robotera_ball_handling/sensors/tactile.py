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
        # Positioned on outer contact surface of the fingertip, pointing outward
        pos=(0.0, 0.008, 0.0),
        rot=(0.7071, 0.7071, 0.0, 0.0),
        convention="world",
    ),
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.5,
        horizontal_aperture=3.0,
        vertical_aperture=3.0,
        clipping_range=(0.002, 0.025),  # 2mm to 25mm contact zone
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
    contact_threshold: float = 0.020,  # 20mm max contact zone threshold
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

    # 1. Zero out non-contact / infinite depth values where no object is in the contact zone
    valid_contact_mask = (tactile_depth > 0.0005) & (tactile_depth < contact_threshold)
    
    # 2. Surface indentation: 0.0 when in open air, positive when an object compresses the pad
    raw_indentation = torch.where(
        valid_contact_mask,
        (contact_threshold - tactile_depth),
        torch.zeros_like(tactile_depth),
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
    num_taxels = output_taxels[0] * output_taxels[1]
    force_array = (indentation * gel_stiffness) / num_taxels

    # 5. If ground-truth physics contact force is supplied, gate and calibrate magnitude
    if contact_forces is not None:
        if contact_forces.ndim >= 1 and contact_forces.shape[-1] == 3:
            contact_forces = torch.norm(contact_forces, dim=-1)
        # Reshape contact forces to align with batch dimensions (*batch_shape, 1, 1)
        c_forces = contact_forces.reshape(*batch_shape, 1, 1)

        in_contact = (c_forces > 0.01)
        taxel_sum = torch.sum(force_array, dim=(-2, -1), keepdim=True)
        
        # When physical contact sensor reports force, scale tactile distribution to match
        scaled_forces = torch.where(
            taxel_sum > 1e-4,
            force_array * (c_forces / taxel_sum.clamp(min=1e-4)),
            torch.full_like(force_array, 1.0 / num_taxels) * c_forces,
        )
        # Strictly zero out forces if physics contact is 0 (open air, no contact with target object)
        force_array = torch.where(in_contact, scaled_forces, torch.zeros_like(force_array))

    return force_array


def compute_visuo_tactile_contact_features(
    tactile_depth: torch.Tensor,
    contact_threshold: float = 0.020,
    gel_stiffness: float = 80.0,
    output_taxels: tuple[int, int] = (8, 8),
    contact_forces: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts both full force array and aggregate fingertip metrics (total force per finger).

    Returns:
        force_array: (num_envs, 5, taxel_h, taxel_w)
        total_forces: (num_envs, 5) total normal force in Newtons per finger
    """
    force_array = compute_visuo_tactile_force_array(
        tactile_depth,
        contact_threshold=contact_threshold,
        gel_stiffness=gel_stiffness,
        output_taxels=output_taxels,
        contact_forces=contact_forces,
    )
    # Sum over spatial taxel dimensions (H, W) to get total normal force per fingertip
    total_forces = torch.sum(force_array, dim=(-2, -1))
    return force_array, total_forces


