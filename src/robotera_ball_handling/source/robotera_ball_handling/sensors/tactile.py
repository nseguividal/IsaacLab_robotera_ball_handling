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
# Modeled after the physical Robotera XHand tactile skin:
# - 12 rows (along longitudinal finger length) x 10 columns (along 270° circumferential wrap)
# - Wide horizontal aperture (3.8) to simulate the 270° cylindrical encirclement around the pad
# - Micro-optical sensing on all 5 fingertips (Thumb, Index, Middle, Ring, Pinky)
# - Macro depth clipping range: 0.1mm to 15mm (0.0001m to 0.015m) for contact surface deformation
# - Efficient 20x24 sensor resolution per fingertip (downsampled 2:1 to 10 cols x 12 rows = 120 taxels)
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
        focal_length=1.2,
        horizontal_aperture=3.8,  # Wide aspect ratio to capture 270° lateral encirclement
        vertical_aperture=3.0,
        clipping_range=(0.0001, 0.015),  # 0.1mm to 15mm continuous tracking range
    ),
    width=20,
    height=24,
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
    output_taxels: tuple[int, int] = (12, 10),  # 12 rows x 10 cols grid (270° encirclement)
    contact_forces: torch.Tensor | None = None,
) -> torch.Tensor:
    """Converts optical visuo-tactile depth maps into a calibrated spatial 2D force array per fingertip.

    Simulates an opaque elastomer gel membrane (Robotera XHand 12x10 tactile skin model):
    - In open air (no physical contact): The elastomer membrane is undeformed (zero force/indentation).
      The camera cannot see through the opaque rubber pad to the palm or other fingers.
    - In physical contact: The object presses into the 3mm elastomer pad.
      * Missed rays (empty air / posinf) represent uncompressed pad (indentation = 0.0mm / Dark Purple).
      * Compressed rays (< 3mm) produce realistic pressure gradients (Red at center -> Yellow -> Blue).
      * Near-clipping penetration (nan / <=0) is treated as maximum indentation (3.0mm / Red).

    Args:
        tactile_depth: Raw depth tensor from visuo-tactile cameras (num_cameras, H, W, 1),
                       (num_cameras, H, W), or (num_envs, 5, H, W).
        contact_threshold: Distance threshold below which contact indentation is registered (gel thickness).
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

    # 1. Determine physical contact state per finger
    if contact_forces is not None:
        if contact_forces.ndim >= 1 and contact_forces.shape[-1] == 3:
            contact_forces = torch.norm(contact_forces, dim=-1)
        c_forces = contact_forces.reshape(*batch_shape, 1, 1)
        in_contact = (c_forces > 0.01)
    else:
        in_contact = None

    # 2. Clean depth:
    # When in contact:
    # - nan (near-clip / penetration < 0.1mm) represents deepest indentation (0.0mm depth)
    # - posinf (ray missed target, empty air > 15mm) represents uncompressed pad (contact_threshold depth -> 0.0 indentation)
    # When in open air: pad is undeformed (contact_threshold depth -> 0.0 indentation)
    if in_contact is not None:
        cleaned_depth = torch.where(
            in_contact,
            torch.nan_to_num(tactile_depth, nan=0.0, posinf=contact_threshold, neginf=0.0),
            torch.full_like(tactile_depth, contact_threshold),
        )
        valid_contact_mask = in_contact & (cleaned_depth >= 0.0) & (cleaned_depth < contact_threshold)
    else:
        cleaned_depth = torch.nan_to_num(tactile_depth, nan=0.0, posinf=contact_threshold, neginf=0.0)
        valid_contact_mask = (cleaned_depth >= 0.0) & (cleaned_depth < contact_threshold)

    # 3. Surface indentation (meters compressed into the elastomer gel)
    raw_indentation = torch.where(
        valid_contact_mask,
        (contact_threshold - cleaned_depth.clamp(min=0.0, max=contact_threshold)),
        torch.zeros_like(cleaned_depth),
    )

    # 4. Downsample to taxel array grid (12 rows x 10 cols force matrix)
    if (h_orig, w_orig) != output_taxels:
        flat_indent = raw_indentation.reshape(-1, 1, h_orig, w_orig)
        pooled = F.adaptive_avg_pool2d(flat_indent, output_taxels)
        indentation = pooled.view(*batch_shape, *output_taxels)
    else:
        indentation = raw_indentation

    # 5. Compute normal force per taxel (Newtons) from optical gel deformation
    force_array = indentation * gel_stiffness

    # 6. If ground-truth physics contact force is supplied, calibrate magnitude
    if in_contact is not None:
        taxel_max = torch.amax(force_array, dim=(-2, -1), keepdim=True)
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



