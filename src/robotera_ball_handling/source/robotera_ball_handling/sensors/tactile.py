# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Visuo-tactile optical sensor configurations modeled after Touchlab & GelSight sensors."""

import math
import torch
import torch.nn.functional as F
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg


# ==============================================================================
# Optical Visuo-Tactile Sensor Array (5 Fingertips - 12x10 Grid with 270° Encirclement)
# ==============================================================================
# Modeled after the Robotera XHand tactile sensor:
# - 12 rows (along longitudinal finger length) x 10 columns (along 270° circumferential wrap)
# - Wide horizontal aperture (3.8) to simulate the 270° cylindrical encirclement around the pad
# - Micro-optical sensing on all 5 fingertips (Thumb, Index, Middle, Ring, Pinky)
# - Macro depth clipping range: 0.1mm to 5mm (0.00001m to 0.005m) for contact surface deformation
# - Efficient 20x24 sensor resolution per fingertip (downsampled 2:1 to 10 cols x 12 rows = 120 taxels)
# ==============================================================================

# Legacy / Unified 5-fingertip camera configuration
VISUO_TACTILE_FINGERTIPS_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/right_hand_.*_tip/visuo_tactile",
    offset=TiledCameraCfg.OffsetCfg(
        # Positioned exactly at the fingertip center facing outward (+Y)
        pos=(0.0, -0.056, 0.002),
        rot=(0.0, 0.7071, 0.7071, 0.0),  # Forward: +Y (outward), Up: -Z (distal tip), Left: +X (thumb)
        convention="world",
    ),
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.2,
        horizontal_aperture=3.8,  # Wide aspect ratio to capture 270° lateral encirclement
        vertical_aperture=3.0,
        clipping_range=(0.00001, 0.005),  # 0.1mm to 5mm continuous tracking range
    ),
    width=20,
    height=24,
)

# ------------------------------------------------------------------------------
# 1. Index Finger Tactile Camera
# ------------------------------------------------------------------------------
VISUO_TACTILE_INDEX_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*right_hand_index_rota_tip/visuo_tactile",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.0, 0.006, 0.002),
        rot=(0.0, 0.7071, 0.7071, 0.0),
        convention="world",
    ),
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.2,
        horizontal_aperture=3.8,
        vertical_aperture=3.0,
        clipping_range=(0.001, 0.01),
    ),
    width=20,
    height=24,
)

# ------------------------------------------------------------------------------
# 2. Middle Finger Tactile Camera
# ------------------------------------------------------------------------------
VISUO_TACTILE_MID_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*right_hand_mid_tip/visuo_tactile",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.0, 0.006, 0.002),
        rot=(0.0, 0.7071, 0.7071, 0.0),
        convention="world",
    ),
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.2,
        horizontal_aperture=3.8,
        vertical_aperture=3.0,
        clipping_range=(0.00001, 0.01),
    ),
    width=20,
    height=24,
)

# ------------------------------------------------------------------------------
# 3. Pinky Finger Tactile Camera
# ------------------------------------------------------------------------------
VISUO_TACTILE_PINKY_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*right_hand_pinky_tip/visuo_tactile",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.0, 0.006, 0.002),
        rot=(0.0, 0.7071, 0.7071, 0.0),
        convention="world",
    ),
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.2,
        horizontal_aperture=3.8,
        vertical_aperture=3.0,
        clipping_range=(0.00001, 0.005),
    ),
    width=20,
    height=24,
)

# ------------------------------------------------------------------------------
# 4. Ring Finger Tactile Camera
# ------------------------------------------------------------------------------
VISUO_TACTILE_RING_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*right_hand_ring_tip/visuo_tactile",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.0, 0.006, 0.002),
        rot=(0.0, 0.7071, 0.7071, 0.0),
        convention="world",
    ),
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.2,
        horizontal_aperture=3.8,
        vertical_aperture=3.0,
        clipping_range=(0.00001, 0.005),
    ),
    width=20,
    height=24,
)

# ------------------------------------------------------------------------------
# 5. Thumb Tactile Camera (TUNE INDEPENDENTLY FOR THUMB)
# ------------------------------------------------------------------------------
# In robotera_xhand_right.urdf, right_hand_thumb_rota_joint3 has rpy="-1.570796 0 1.570796".
# Tune pos=(x, y, z) and rot=(w, x, y, z) here without affecting any other finger:
VISUO_TACTILE_THUMB_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*right_hand_thumb_rota_tip/visuo_tactile",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.0, 0.01, 0.001),           
        rot=(0.0, 0.7071, 0.7071, 0.0),     
        convention="world",
    ),
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.2,
        horizontal_aperture=3.8,
        vertical_aperture=3.0,
        clipping_range=(0.00001, 0.015),
    ),
    width=20,
    height=24,
)
# ==============================================================================
# Fingertip Contact Force & Pressure Reporter (Physics Sensor)
# ==============================================================================
# Used just to check if the fingertip is in contact with an object, to debug the image of the visuotactile sensor
# ==============================================================================

FINGERTIP_CONTACT_SENSOR_CFG = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/right_hand_.*_link2",           # fingertip link
    history_length=3,
    update_period=0.0,
    track_air_time=False,
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball1", "{ENV_REGEX_NS}/Ball2"],
)


# ==============================================================================
# Helper to read tactile depth tensor from scene
# ==============================================================================
def get_tactile_depth_tensor(scene) -> torch.Tensor:
    """Retrieves and consolidates tactile depth tensors across all 5 fingertips.

    Supports both:
    1. Independent per-finger cameras:
       tactile_cam_index, tactile_cam_mid, tactile_cam_pinky, tactile_cam_ring, tactile_cam_thumb
    2. Legacy unified multi-camera array:
       tactile_cameras

    The returned tensor is strictly sorted in PhysX alphabetical body order:
    [Index, Middle, Pinky, Ring, Thumb], with shape (num_envs, 5, H, W) or (5, H, W).
    """
    if hasattr(scene, "sensors") and "tactile_cam_index" in scene.sensors:
        cam_keys = [
            "tactile_cam_index",
            "tactile_cam_mid",
            "tactile_cam_pinky",
            "tactile_cam_ring",
            "tactile_cam_thumb",
        ]
        depth_list = []
        for k in cam_keys:
            d = scene[k].data.output["distance_to_image_plane"]
            if d.ndim >= 3 and d.shape[-1] == 1:
                d = d.squeeze(-1)
            depth_list.append(d)
        if depth_list[0].ndim == 2:
            return torch.stack(depth_list, dim=0)
        else:
            return torch.stack(depth_list, dim=1)
    else:
        return scene["tactile_cameras"].data.output["distance_to_image_plane"]


# ==============================================================================
# Visuo-Tactile to Spatial Force Array Converter
# ==============================================================================
def compute_visuo_tactile_force_array(
    tactile_depth: torch.Tensor,
    contact_threshold: float = 0.015,  # 15mm continuous tracking / contact threshold
    gel_stiffness: float = 80.0,       # Effective elasticity scaling (N/m)
    output_taxels: tuple[int, int] = (12, 10),  # 12 rows x 10 cols grid (270° encirclement)
    contact_forces: torch.Tensor | None = None,
) -> torch.Tensor:
    """Converts optical visuo-tactile depth maps into a calibrated spatial 2D force array per fingertip.

    Directly measures distance/depth from the camera to the target object surface:
    - In open air: Tracks the approaching object continuously from 15mm away down to contact,
      displaying the natural spherical curvature as concentric proximity contours.
    - In contact: Physical normal forces calibrate the distribution magnitude across active taxels.

    Args:
        tactile_depth: Raw depth tensor from visuo-tactile cameras (num_cameras, H, W, 1),
                       (num_cameras, H, W), or (num_envs, 5, H, W).
        contact_threshold: Distance threshold below which proximity/indentation is registered.
        gel_stiffness: Effective elastic stiffness scaling (N/m).
        output_taxels: Spatial taxel grid resolution (12 rows x 10 cols matrix per fingertip).
        contact_forces: Optional direct physics normal forces (num_envs, 5) or (num_cameras,) for calibration.

    Returns:
        torch.Tensor: Force array in Newtons per taxel of shape (*batch_dims, taxel_h, taxel_w).
    """
    # 0. Squeeze trailing channel dimension if present (e.g. [N, H, W, 1] -> [N, H, W])
    if tactile_depth.ndim >= 3 and tactile_depth.shape[-1] == 1:
        tactile_depth = tactile_depth.squeeze(-1)

    batch_shape = tactile_depth.shape[:-2]
    h_orig, w_orig = tactile_depth.shape[-2], tactile_depth.shape[-1]

    # 1. Determine physical contact state per finger if forces supplied
    if contact_forces is not None:
        if contact_forces.ndim >= 1 and contact_forces.shape[-1] == 3:
            contact_forces = torch.norm(contact_forces, dim=-1)
        c_forces = contact_forces.reshape(*batch_shape, 1, 1)
        in_contact = (c_forces > 0.01)
        # In open air: nan / posinf are empty air -> contact_threshold (zero indentation)
        # In contact: nan / near-clip (< 0.1mm) is deepest penetration -> 0.0mm depth
        cleaned_depth = torch.where(
            in_contact,
            torch.nan_to_num(tactile_depth, nan=0.0, posinf=contact_threshold, neginf=0.0),
            torch.nan_to_num(tactile_depth, nan=contact_threshold, posinf=contact_threshold, neginf=contact_threshold),
        )
    else:
        in_contact = None
        cleaned_depth = torch.nan_to_num(tactile_depth, nan=contact_threshold, posinf=contact_threshold, neginf=contact_threshold)

    cleaned_depth = torch.clamp(cleaned_depth, min=0.0, max=contact_threshold)

    # 2. Continuous optical proximity / indentation: Maximum at 0.0m, 0.0 at contact_threshold
    raw_indentation = contact_threshold - cleaned_depth

    # 3. Downsample to taxel array grid (12 rows x 10 cols force matrix)
    if (h_orig, w_orig) != output_taxels:
        flat_indent = raw_indentation.reshape(-1, 1, h_orig, w_orig)
        pooled = F.adaptive_avg_pool2d(flat_indent, output_taxels)
        indentation = pooled.view(*batch_shape, *output_taxels)
    else:
        indentation = raw_indentation

    # 4. Compute normal force per taxel (Newtons) from optical proximity/indentation
    force_array = indentation * gel_stiffness

    # 5. If ground-truth physics contact force is supplied, gate and calibrate magnitude
    if in_contact is not None:
        taxel_max = torch.amax(force_array, dim=(-2, -1), keepdim=True)
        scaled_forces = torch.where(
            taxel_max > 1e-4,
            force_array * (c_forces / taxel_max.clamp(min=1e-4)),
            torch.full_like(force_array, 1.0) * c_forces,
        )
        # When physical contact sensor reports force, scale tactile distribution so peak square = peak contact force
        # When in air, keep zero physical force
        force_array = torch.where(in_contact, scaled_forces, torch.zeros_like(force_array))

    return force_array


def compute_visuo_tactile_contact_features(
    tactile_depth: torch.Tensor,
    contact_threshold: float = 0.015,
    gel_stiffness: float = 80.0,
    output_taxels: tuple[int, int] = (12, 10),
    contact_forces: torch.Tensor | None = None,
    dynamics_model: "TactileSignalDynamics | None" = None,
    dt: float = 1.0 / 60.0,
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
    if dynamics_model is not None:
        force_array = dynamics_model.apply(force_array, dt=dt)

    # Maximum over spatial taxel dimensions (H, W) to get the total peak contact force felt by the finger
    total_forces = torch.amax(force_array, dim=(-2, -1))
    return force_array, total_forces


class TactileSignalDynamics:
    """Simulates viscoelastic creep decay, baseline drift, and sensor noise for elastomeric pads.

    Physical phenomena modeled:
    1. Maxwell/Kelvin-Voigt viscoelastic stress relaxation:
       When held under continuous compression, the elastomer silicone relaxes over time,
       causing perceived indentation force to decay slightly (5% - 12%) over a time constant tau (~2.0s).
    2. Zero-baseline random walk / thermal drift:
       Temperature variation and sensor ADC baseline shifts cause slow random baseline wander (+/- 0.02 N).
    3. Analog taxel jitter / noise:
       High-frequency Gaussian noise on individual taxels (sigma = 0.004 N).
    """

    def __init__(
        self,
        decay_ratio: float = 0.08,
        relaxation_time_s: float = 2.0,
        drift_rate: float = 0.003,
        noise_std: float = 0.004,
    ):
        self.decay_ratio = decay_ratio
        self.relaxation_time_s = relaxation_time_s
        self.drift_rate = drift_rate
        self.noise_std = noise_std
        self.history_force: torch.Tensor | None = None
        self.baseline_drift: torch.Tensor | None = None

    def reset(self):
        self.history_force = None
        self.baseline_drift = None

    def apply(self, forces: torch.Tensor, dt: float = 1.0 / 60.0) -> torch.Tensor:
        """Applies temporal viscoelastic relaxation, drift, and noise to the force tensor."""
        if self.history_force is None or self.history_force.shape != forces.shape:
            self.history_force = forces.clone()
            self.baseline_drift = torch.zeros_like(forces)

        # 1. Viscoelastic relaxation under continuous sustained load
        alpha = math.exp(-dt / max(self.relaxation_time_s, 1e-3))
        sustained = torch.minimum(forces, self.history_force)
        relaxed_force = forces - self.decay_ratio * (1.0 - alpha) * sustained
        self.history_force = relaxed_force.detach()

        # 2. Baseline random walk drift (mean-reverting Ornstein-Uhlenbeck process)
        reversion_rate = 0.99
        wander = torch.randn_like(forces) * self.drift_rate
        self.baseline_drift = self.baseline_drift * reversion_rate + wander
        self.baseline_drift.clamp_(-0.025, 0.025)

        # 3. High-frequency analog sensor noise
        noise = torch.randn_like(forces) * self.noise_std

        # Only apply noise and drift where force is non-zero, or add subtle baseline noise
        output = torch.where(forces > 0.0, torch.clamp(relaxed_force + self.baseline_drift + noise, min=0.0), torch.zeros_like(forces))
        return output



