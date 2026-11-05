# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Camera sensor configurations modeled after Intel RealSense cameras."""

import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg


# ==============================================================================
# Intel RealSense D405 - Wrist-Mounted Sub-Millimeter Macro Camera
# ==============================================================================
# The D405 is Intel's sub-millimeter close-range RGB-D camera designed for
# robotic wrists and high-precision dexterous in-hand manipulation.
# Official RealSense D405 Datasheet Specs:
# - RGB Frame Resolution: 1280 x 720 (at 30 fps, up to 90 fps)
# - Depth Output Resolution: Up to 1280 x 720 (up to 90 fps)
# - Ideal Range: 7 cm to 50 cm (0.07 m to 0.50 m)
# - Minimum Depth Distance (Min-Z): 7 cm (0.07 m)
# - Depth & RGB Field of View (FOV): 87° (H) x 58° (V)
# - Sensor: Global Shutter RGB with ISP on Left Imager
# - Optics: focal length 1.93 mm, horizontal aperture 3.663 mm, vertical aperture 2.140 mm
# ==============================================================================

REALSENSE_D405_WRIST_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/realsense_d405_wrist",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.06, 0.0, 0.5),
        # Set so that the UI displays exactly (0.627, 0.326, -0.326, -0.627)
        rot=(0.95372, 0.0, 0.30070, 0.0),
        convention="world",
    ),
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.93,
        horizontal_aperture=3.663,  # HFOV = 2 * atan(3.663 / (2 * 1.93)) = 87.0°
        vertical_aperture=2.140,    # VFOV = 2 * atan(2.140 / (2 * 1.93)) = 58.0°
        clipping_range=(0.07, 0.50),  # Exact D405 ideal range: 7 cm to 50 cm
    ),
    width=1280,
    height=720,
)


# ==============================================================================
# Intel RealSense D435 - Overview / Tabletop Camera
# ==============================================================================
# The D435 is a wide-field RGB-D camera providing global scene context.
# Official RealSense D435 Datasheet Specs:
# - Depth Output Resolution: Up to 1280 x 720 (up to 90 fps)
# - RGB Frame Resolution: Up to 1280 x 800 (1280 x 720 at 60 fps)
# - Ideal Range: 0.3 m to 3.0 m
# - Minimum Depth Distance (Min-Z): 20 cm (0.20 m)
# - Depth Field of View (FOV): 87° (H) x 58° (V)
# - RGB Sensor FOV: 90° (H) x 65° (V) (90° x 58.7° at 16:9 1280x720)
# - Optics: focal length 1.93 mm, horizontal aperture 3.860 mm (90.0° HFOV), vertical aperture 2.171 mm (58.7° VFOV)
# ==============================================================================

REALSENSE_D435_OVERVIEW_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/realsense_d435_overview",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.15, 0.2, 0.85),
        # Set to produce (0, 0, 0.3, 0.95) in the UI
        rot=(0.6275, 0.326, 0.326, -0.6275),
        convention="world",
    ),
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.93,
        horizontal_aperture=3.860,  # HFOV = 2 * atan(3.860 / (2 * 1.93)) = 90.0° (RGB wide FOV)
        vertical_aperture=2.171,    # VFOV = 2 * atan(2.171 / (2 * 1.93)) = 58.7° (16:9 aspect ratio)
        clipping_range=(0.20, 3.00),  # D435 Min-Z (20 cm) to Ideal Far Range (3.0 m)
    ),
    width=1280,
    height=720,
)


# ==============================================================================
# 4 Overhead Ceiling Cameras - Multi-View Room Teleoperation & Surveillance
# ==============================================================================
# 4 synchronized cameras mounted on the room ceiling (Z ≈ 2.40m) overlooking
# the workspace table at (0.20, 0.0, 0.35) from 4 distinct corner perspectives:
# 1. Front-Left  (North-West): pos=(0.80, -0.60, 2.40)
# 2. Front-Right (North-East): pos=(0.80,  0.60, 2.40)
# 3. Rear-Left   (South-West): pos=(-0.40, -0.60, 2.40)
# 4. Rear-Right  (South-East): pos=(-0.40,  0.60, 2.40)
# Resolution: 1280 x 720 (16:9), ROS convention (+Z forward, -Y up)
# ==============================================================================

CEILING_CAMERA_FRONT_LEFT_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/ceiling_cam_front_left",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.80, -0.60, 2.40),
        rot=(-0.1801, 0.9062, 0.3753, -0.0746),
        convention="ros",
    ),
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=2.50,
        horizontal_aperture=3.860,
        vertical_aperture=2.171,
        clipping_range=(0.20, 5.00),
    ),
    width=1280,
    height=720,
)

CEILING_CAMERA_FRONT_RIGHT_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/ceiling_cam_front_right",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.80, 0.60, 2.40),
        rot=(-0.0746, 0.3753, 0.9062, -0.1801),
        convention="ros",
    ),
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=2.50,
        horizontal_aperture=3.860,
        vertical_aperture=2.171,
        clipping_range=(0.20, 5.00),
    ),
    width=1280,
    height=720,
)

CEILING_CAMERA_REAR_LEFT_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/ceiling_cam_rear_left",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(-0.40, -0.60, 2.40),
        rot=(-0.1801, 0.9062, -0.3753, 0.0746),
        convention="ros",
    ),
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=2.50,
        horizontal_aperture=3.860,
        vertical_aperture=2.171,
        clipping_range=(0.20, 5.00),
    ),
    width=1280,
    height=720,
)

CEILING_CAMERA_REAR_RIGHT_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/ceiling_cam_rear_right",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(-0.40, 0.60, 2.40),
        rot=(0.0746, -0.3753, 0.9062, -0.1801),
        convention="ros",
    ),
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=2.50,
        horizontal_aperture=3.860,
        vertical_aperture=2.171,
        clipping_range=(0.20, 5.00),
    ),
    width=1280,
    height=720,
)


# ==============================================================================
# Helper Functions: Spatially Correlated Low-Frequency Noise (Technique 3)
# ==============================================================================

def _generate_correlated_noise(
    shape: tuple[int, ...],
    cluster_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Generates Gaussian noise (single-pixel crisp when cluster_size<=1, low-pass patches when >1)."""
    if cluster_size <= 1:
        return torch.randn(shape, device=device)
    b, c, h, w = shape
    h_low = max(2, h // cluster_size)
    w_low = max(2, w // cluster_size)
    low_res = torch.randn((b, c, h_low, w_low), device=device)
    smooth = F.interpolate(low_res, size=(h, w), mode="bilinear", align_corners=False)
    return smooth / (smooth.std() + 1e-6)


def _generate_correlated_uniform(
    shape: tuple[int, ...],
    cluster_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Generates uniform noise in [0, 1] (single-pixel crisp when cluster_size<=1, continuous when >1)."""
    if cluster_size <= 1:
        return torch.rand(shape, device=device)
    b, c, h, w = shape
    h_low = max(2, h // cluster_size)
    w_low = max(2, w // cluster_size)
    low_res = torch.rand((b, c, h_low, w_low), device=device)
    smooth = F.interpolate(low_res, size=(h, w), mode="bilinear", align_corners=False)
    return smooth.clamp(0.0, 1.0)


# ==============================================================================
# Physics-Based RealSense D405 Stereo Noise & Artifact Model
# ==============================================================================

class RealSenseD405NoiseModel:
    """Physics-based RealSense D405 stereoscopic camera noise and artifact simulator.

    Simulates the physical optical and ASIC phenomena of the Intel RealSense D405:
    1. Low-Contrast / Textureless Stereo Dropout (Zero-depth holes)
    2. Stereo Occlusion Shadows (stereo baseline blind spots on silhouettes)
    3. Saturation Dropout
    4. Edge Flickering (Stereo correlation window bleeding)
    5. Hard Optical Min-Z (7 cm) and Max-Z (50 cm) Search Limits
    6. RealSense Spatial Filter & Inter-Finger Depth Fading
    """

    # Module-level Master Filter Flags (toggle here to activate/deactivate globally)
    DEFAULT_ENABLE_LOW_CONTRAST: bool = True
    DEFAULT_CONTRAST_THRESHOLD: float = 0.0013  # Sobel gradient cutoff: regions with gradient < threshold drop to 0.0 (black) // close to 0.0 = ideal behaviour
    DEFAULT_ENABLE_STEREO_OCCLUSION: bool = True
    DEFAULT_ENABLE_VORONOI_DROPOUT: bool = False
    DEFAULT_ENABLE_EDGE_FLICKER: bool = True                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    DEFAULT_ENABLE_SPATIAL_FILTER: bool = False  # RealSense Edge-Preserving Spatial Filter 
    DEFAULT_OCCLUSION_SIDE: str = "left"  # "left" (matches real-world wrist D405 orientation), "right", or "both"

    # Edge flicker tuning defaults (can be tuned right here at class level)
    DEFAULT_DEPTH_NOISE_STD: float = 0.0030
    DEFAULT_EDGE_FLICKER_PROB: float = 0.40
    DEFAULT_EDGE_FLICKER_CLUSTER_SIZE: int = 1   # 1 = crisp single pixel noise (fine grain)
    DEFAULT_EDGE_HALO_RADIUS: int = 5            # Halo boundary reach in pixels
    DEFAULT_EDGE_FLICKER_DEPTH_OFFSET: float = 0.015

    # Spatial filter tuning defaults
    DEFAULT_SPATIAL_FILTER_RADIUS: int = 4
    DEFAULT_SPATIAL_DELTA_THRESHOLD: float = 0.040
    DEFAULT_ENABLE_SPATIAL_HOLE_FILLING: bool = True
    DEFAULT_SPATIAL_HOLE_FILL_RADIUS: int = 14
    DEFAULT_SPATIAL_HOLE_FILL_OFFSET: float = 0.030

    def __init__(
        self,
        baseline_m: float = 0.018,
        min_depth: float = 0.07,
        max_depth: float = 0.30,
        subpixel_levels: int = 32,
        contrast_threshold: float | None = None,     # Between: 0.00 (no low-contrast dropouts) to ~0.50 (drops even sharp edges)
        noise_std_at_50cm: float = 0.010,
        # --- Sim-to-Real Filter Activation Flags ---
        enable_low_contrast: bool | None = None,
        enable_voronoi_dropout: bool | None = None,
        enable_edge_flicker: bool | None = None,
        enable_spatial_filter: bool | None = None,
        enable_stereo_occlusion: bool | None = None,
        occlusion_side: str | None = None,
        # --- Filter Tuning Parameters ---
        depth_noise_std: float | None = None,
        edge_flicker_prob: float | None = None,
        edge_flicker_cluster_size: int | None = None,
        edge_halo_radius: int | None = None,
        edge_flicker_depth_offset: float | None = None,
        # --- RealSense Spatial Filter Parameters (Technique 4) ---
        spatial_filter_radius: int | None = None,
        spatial_delta_threshold: float | None = None,
        enable_spatial_hole_filling: bool | None = None,
        spatial_hole_fill_radius: int | None = None,
        spatial_hole_fill_offset: float | None = None,
        # --- Voronoi Cellular Low-Contrast Dropout Parameters ---
        voronoi_cell_size: int = 80,
        voronoi_spot_radius: float = 20.0,
        voronoi_density: float = 0.05,
        voronoi_contrast_threshold: float = 0.001,
        # --- Additional Features 
        enable_flying_pixels: bool = False,
        enable_specular_dropout: bool = False,
    ):
        self.baseline_m = baseline_m
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.subpixel_levels = subpixel_levels
        self.contrast_threshold = (
            self.DEFAULT_CONTRAST_THRESHOLD if contrast_threshold is None else contrast_threshold
        )
        self.noise_std_at_50cm = noise_std_at_50cm
        self.enable_low_contrast = (
            self.DEFAULT_ENABLE_LOW_CONTRAST if enable_low_contrast is None else enable_low_contrast
        )
        self.enable_voronoi_dropout = (
            self.DEFAULT_ENABLE_VORONOI_DROPOUT if enable_voronoi_dropout is None else enable_voronoi_dropout
        )
        self.enable_edge_flicker = (
            self.DEFAULT_ENABLE_EDGE_FLICKER if enable_edge_flicker is None else enable_edge_flicker
        )
        self.enable_spatial_filter = (
            self.DEFAULT_ENABLE_SPATIAL_FILTER if enable_spatial_filter is None else enable_spatial_filter
        )
        self.enable_stereo_occlusion = (
            self.DEFAULT_ENABLE_STEREO_OCCLUSION
            if enable_stereo_occlusion is None
            else enable_stereo_occlusion
        )
        self.occlusion_side = (
            self.DEFAULT_OCCLUSION_SIDE
            if occlusion_side is None
            else occlusion_side
        )
        self.depth_noise_std = (
            self.DEFAULT_DEPTH_NOISE_STD if depth_noise_std is None else depth_noise_std
        )
        self.edge_flicker_prob = (
            self.DEFAULT_EDGE_FLICKER_PROB if edge_flicker_prob is None else edge_flicker_prob
        )
        self.edge_flicker_cluster_size = (
            self.DEFAULT_EDGE_FLICKER_CLUSTER_SIZE if edge_flicker_cluster_size is None else edge_flicker_cluster_size
        )
        self.edge_halo_radius = (
            self.DEFAULT_EDGE_HALO_RADIUS if edge_halo_radius is None else edge_halo_radius
        )
        self.edge_flicker_depth_offset = (
            self.DEFAULT_EDGE_FLICKER_DEPTH_OFFSET if edge_flicker_depth_offset is None else edge_flicker_depth_offset
        )
        self.spatial_filter_radius = (
            self.DEFAULT_SPATIAL_FILTER_RADIUS if spatial_filter_radius is None else spatial_filter_radius
        )
        self.spatial_delta_threshold = (
            self.DEFAULT_SPATIAL_DELTA_THRESHOLD if spatial_delta_threshold is None else spatial_delta_threshold
        )
        self.enable_spatial_hole_filling = (
            self.DEFAULT_ENABLE_SPATIAL_HOLE_FILLING if enable_spatial_hole_filling is None else enable_spatial_hole_filling
        )
        self.spatial_hole_fill_radius = (
            self.DEFAULT_SPATIAL_HOLE_FILL_RADIUS if spatial_hole_fill_radius is None else spatial_hole_fill_radius
        )
        self.spatial_hole_fill_offset = (
            self.DEFAULT_SPATIAL_HOLE_FILL_OFFSET if spatial_hole_fill_offset is None else spatial_hole_fill_offset
        )
        self.voronoi_cell_size = voronoi_cell_size
        self.voronoi_spot_radius = voronoi_spot_radius
        self.voronoi_density = voronoi_density
        self.voronoi_contrast_threshold = voronoi_contrast_threshold
        self.enable_flying_pixels = enable_flying_pixels
        self.enable_specular_dropout = enable_specular_dropout

    def __call__(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Applies physical D405 stereo noise and artifacts to ground-truth depth.

        Args:
            rgb: Rendered RGB/RGBA image tensor of shape (..., H, W, 3/4) in [0, 1] or [0, 255].
            depth: Ground-truth distance-to-image-plane depth tensor of shape (..., H, W) or (..., H, W, 1) in meters.

        Returns:
            Noisy RealSense depth map of shape matching input depth with realistic dropouts (0.0 for invalid).
        """
        orig_shape = depth.shape
        device = depth.device

        # Reshape depth to 4D: (B, 1, H, W)
        if depth.ndim == 2:
            d = depth.unsqueeze(0).unsqueeze(0)
            rgb_in = rgb.unsqueeze(0)
        elif depth.ndim == 3:
            if depth.shape[-1] == 1:
                d = depth.squeeze(-1).unsqueeze(0).unsqueeze(0)
                rgb_in = rgb.unsqueeze(0)
            else:
                d = depth.unsqueeze(1)
                rgb_in = rgb
        elif depth.ndim == 4:
            if depth.shape[-1] == 1:
                d = depth.squeeze(-1).unsqueeze(1)
            else:
                d = depth
            rgb_in = rgb
        else:
            raise ValueError(f"Unsupported depth shape: {depth.shape}")

        b, _, h, w = d.shape

        # Compute focal length in pixels for D405 (HFOV = 87 deg)
        # HFOV = 2 * atan(W / (2 * fx)) => fx = W / (2 * tan(43.5 deg))
        fx = w / (2.0 * 0.948965)
        fb = fx * self.baseline_m

        # Convert RGB to normalized grayscale luminance [0.0, 1.0]
        if rgb_in.shape[-1] == 4:
            rgb_rgb = rgb_in[..., :3]
        else:
            rgb_rgb = rgb_in

        if rgb_rgb.max() > 1.0:
            rgb_norm = rgb_rgb.float() / 255.0
        else:
            rgb_norm = rgb_rgb.float()

        # (B, H, W, 3) -> (B, 1, H, W)
        gray = (0.299 * rgb_norm[..., 0] + 0.587 * rgb_norm[..., 1] + 0.114 * rgb_norm[..., 2]).view(b, 1, h, w)

        # Standard 3x3 Sobel kernels
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=device).view(1, 1, 3, 3) / 8.0

        # Initialize noisy depth from ground truth
        noisy_d = d.clone()

        # ----------------------------------------------------------------------
        # 1. Optical Range Invalidation (D405 Hardware Cutoff: > 50cm or < 7cm)
        # ----------------------------------------------------------------------
        # On real RealSense D405, objects farther than 50cm (0.50m) or closer
        # than 7cm (0.07m) have zero disparity and output 0.0 (appear black).
        # We apply this FIRST so the background (>50cm) becomes empty space (0.0).
        range_invalid = (d > 0.50) | (d < self.min_depth)
        noisy_d[range_invalid] = 0.0

        # ----------------------------------------------------------------------
        # 2. LOW-CONTRAST Dropout (Sobel Gradient on Luminance)
        # ----------------------------------------------------------------------
        local_contrast = None
        if self.enable_low_contrast or self.enable_voronoi_dropout:
            grad_x = F.conv2d(gray, sobel_x, padding=1)
            grad_y = F.conv2d(gray, sobel_y, padding=1)
            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
            local_contrast = F.max_pool2d(grad_mag, kernel_size=5, stride=1, padding=2)

        if self.enable_low_contrast and local_contrast is not None:
            # Full low-contrast dropout: zero out entire featureless surfaces
            low_contrast_dropout = local_contrast < self.contrast_threshold
            noisy_d[low_contrast_dropout] = 0.0

        # ----------------------------------------------------------------------
        # 2b. Voronoi / Cellular Low-Contrast Dropout (Spotty 0-depth holes)
        # ----------------------------------------------------------------------
        # Simulates real-world stereo camera behavior on smooth/low-texture surfaces:
        # Instead of dropping out the entire surface into a solid black void, stereo
        # matching fails in an organic Voronoi distribution of discrete black
        # patches/holes (0 depth), leaving islands of valid depth between them.
        if self.enable_voronoi_dropout and local_contrast is not None:
            low_contrast_region = (local_contrast < self.voronoi_contrast_threshold) & (noisy_d > 0.0)

            # Generate 2D Voronoi cellular grid
            S = self.voronoi_cell_size
            H, W = noisy_d.shape[-2], noisy_d.shape[-1]
            B = noisy_d.shape[0] if noisy_d.ndim >= 3 else 1
            gh = (H + S - 1) // S + 2
            gw = (W + S - 1) // S + 2

            # Random seed jitter within each grid cell
            jitter_x = torch.rand((gh, gw), device=device)
            jitter_y = torch.rand((gh, gw), device=device)
            grid_c = torch.arange(gw, device=device).view(1, gw).float()
            grid_r = torch.arange(gh, device=device).view(gh, 1).float()
            seed_x = (grid_c + jitter_x) * S
            seed_y = (grid_r + jitter_y) * S

            # Randomly decide which cells drop out
            cell_active = (torch.rand((gh, gw), device=device) < self.voronoi_density)

            # Pixel coordinate grids
            y_coords = torch.arange(H, device=device).view(H, 1).float()
            x_coords = torch.arange(W, device=device).view(1, W).float()
            cy = (y_coords / S).long() + 1
            cx = (x_coords / S).long() + 1

            min_d2 = torch.full((H, W), 1e9, device=device)
            active_at_min = torch.zeros((H, W), dtype=torch.bool, device=device)

            # Check 9 neighboring Voronoi cells
            for dy in (-1, 0, 1):
                ny = torch.clamp(cy + dy, 0, gh - 1)
                for dx in (-1, 0, 1):
                    nx = torch.clamp(cx + dx, 0, gw - 1)
                    sx = seed_x[ny, nx]
                    sy = seed_y[ny, nx]
                    act = cell_active[ny, nx]

                    d2 = (x_coords - sx) ** 2 + (y_coords - sy) ** 2
                    is_closer = d2 < min_d2
                    min_d2 = torch.where(is_closer, d2, min_d2)
                    active_at_min = torch.where(is_closer, act, active_at_min)

            # In low-contrast regions, drop out pixels within the Voronoi spot radius
            min_dist = torch.sqrt(min_d2).view(1, 1, H, W).expand(B, 1, H, W)
            active_mask = active_at_min.view(1, 1, H, W).expand(B, 1, H, W)

            # Scale spot radius dynamically based on contrast deficit (lower contrast = larger holes)
            contrast_norm = torch.clamp(local_contrast / self.voronoi_contrast_threshold, 0.0, 1.0)
            effective_radius = self.voronoi_spot_radius * (1.5 - 0.5 * contrast_norm)

            voronoi_holes = low_contrast_region & active_mask & (min_dist < effective_radius)
            noisy_d[voronoi_holes] = 0.0

        # ----------------------------------------------------------------------
        # 3. Stereo Parallax Occlusion Shadows (Baseline Blind Spot Gaps)
        # ----------------------------------------------------------------------
        # In stereo vision, foreground objects block the secondary camera from seeing
        # background surfaces immediately adjacent to the foreground silhouette.
        # - "left": Occlusion shadow falls on the LEFT of foreground silhouettes
        #           (matches real-world wrist D405 orientation / secondary lens on the left).
        # - "right": Occlusion shadow falls on the RIGHT of foreground silhouettes.
        # - "both": Bilateral occlusion gaps.
        if self.enable_stereo_occlusion:
            valid_surf = (noisy_d >= self.min_depth) & (noisy_d <= 0.50)
            w_img = noisy_d.shape[-1]
            k_occ = max(4, min(32, int(0.045 * w_img)))

            z_left = noisy_d[:, :, :, :-1]
            z_right = noisy_d[:, :, :, 1:]
            val_left = valid_surf[:, :, :, :-1]
            val_right = valid_surf[:, :, :, 1:]

            occlusion_mask = torch.zeros_like(valid_surf, dtype=torch.bool)

            # --- Left-side Occlusion (Secondary lens to the left / camera inverted) ---
            if self.occlusion_side in ["left", "both"]:
                # Step from deeper background on the left to foreground on the right
                step_to_fg_right = val_left & val_right & ((z_left - z_right) > 0.015)
                # Foreground left boundary (at column of z_right)
                step_edge_left = F.pad(step_to_fg_right, (1, 0, 0, 0), value=False)
                # Pad k_occ on right so 1D horizontal pooling propagates shadow leftward
                padded_step_l = F.pad(step_edge_left.float(), (0, k_occ, 0, 0), value=0.0)
                cand_l = F.max_pool2d(padded_step_l, kernel_size=(1, k_occ + 1), stride=1) > 0.5
                fg_edge_z_l = torch.where(step_edge_left, noisy_d, torch.full_like(noisy_d, 999.0))
                padded_fg_l = F.pad(fg_edge_z_l, (0, k_occ, 0, 0), value=999.0)
                dilated_fg_l = -F.max_pool2d(-padded_fg_l, kernel_size=(1, k_occ + 1), stride=1)
                occ_l = cand_l & valid_surf & (noisy_d > (dilated_fg_l + 0.010))
                occlusion_mask = occlusion_mask | occ_l

            # --- Right-side Occlusion (Secondary lens to the right) ---
            if self.occlusion_side in ["right", "both"]:
                # Step from foreground on the left to deeper background on the right
                step_to_bg_right = val_left & val_right & ((z_right - z_left) > 0.015)
                # Foreground right boundary (at column of z_left)
                step_edge_right = F.pad(step_to_bg_right, (0, 1, 0, 0), value=False)
                # Pad k_occ on left so 1D horizontal pooling propagates shadow rightward
                padded_step_r = F.pad(step_edge_right.float(), (k_occ, 0, 0, 0), value=0.0)
                cand_r = F.max_pool2d(padded_step_r, kernel_size=(1, k_occ + 1), stride=1) > 0.5
                fg_edge_z_r = torch.where(step_edge_right, noisy_d, torch.full_like(noisy_d, 999.0))
                padded_fg_r = F.pad(fg_edge_z_r, (k_occ, 0, 0, 0), value=999.0)
                dilated_fg_r = -F.max_pool2d(-padded_fg_r, kernel_size=(1, k_occ + 1), stride=1)
                occ_r = cand_r & valid_surf & (noisy_d > (dilated_fg_r + 0.010))
                occlusion_mask = occlusion_mask | occ_r

            # Zero out occluded pixels (RealSense lost-lock stereo gap)
            noisy_d = torch.where(occlusion_mask, torch.zeros_like(noisy_d), noisy_d)

        # ----------------------------------------------------------------------
        # 4. Solid Surface Depth Noise & Boundary Edge Flickering (Technique 3)
        # ----------------------------------------------------------------------
        # Applied AFTER the 50cm range filter so that boundary edge flickering
        # and noisy flying pixels can bleed into the surrounding empty space / background.
        # Simulates real-world D405 characteristics:
        # a) Surface noise: Spatially correlated low-frequency axial noise across
        #    the finger and ball surfaces so they have natural smooth waviness.
        # b) Silhouette halo: Dilation expands foreground across narrow finger gaps.
        # c) Spatially correlated noise: Instead of 1x1 salt-and-pepper grain,
        #    flickering occurs as continuous, organic patches across a wider surface.
        # d) Inter-finger depth recess: Depth in the halo is slightly deeper (+2cm),
        #    causing it to render in warm/red tones uniting the fingers.
        if self.enable_edge_flicker:
            valid_depth = (noisy_d >= self.min_depth) & (noisy_d <= self.max_depth)

            # 4a. Surface depth noise: spatially correlated Gaussian axial variation on valid surfaces
            axial_noise = _generate_correlated_noise(noisy_d.shape, self.edge_flicker_cluster_size, device)
            axial_jitter = axial_noise * self.depth_noise_std
            noisy_d = torch.where(valid_depth, noisy_d + axial_jitter, noisy_d)

            # 4b. Internal step edge noise: disparity jitter across depth discontinuities
            d_grad_x = F.conv2d(noisy_d, sobel_x, padding=1)
            d_grad_y = F.conv2d(noisy_d, sobel_y, padding=1)
            step_edges = (torch.sqrt(d_grad_x ** 2 + d_grad_y ** 2) > 0.02) & valid_depth
            edge_jitter = _generate_correlated_noise(noisy_d.shape, self.edge_flicker_cluster_size, device) * 0.010
            noisy_d = torch.where(step_edges, noisy_d + edge_jitter, noisy_d)

            # 4c. Silhouette halo: expand valid foreground mask into adjacent empty/background pixels
            # A wider kernel (e.g. 15px) allows the halo to bridge across narrow inter-finger spaces
            k_halo = max(3, self.edge_halo_radius if self.edge_halo_radius % 2 == 1 else self.edge_halo_radius + 1)
            pad_halo = k_halo // 2
            dilated_mask = F.max_pool2d(valid_depth.float(), kernel_size=k_halo, stride=1, padding=pad_halo) > 0.5
            halo_mask = dilated_mask & (~valid_depth)

            # Correlated uniform noise forms smooth continuous clusters/patches instead of 1x1 pixels
            smooth_u = _generate_correlated_uniform(noisy_d.shape, self.edge_flicker_cluster_size, device)
            flicker_active = (smooth_u < self.edge_flicker_prob) & halo_mask

            # Sample adjacent FOREGROUND finger depth (mask background to +999 and min-pool)
            fg_depth = torch.where(valid_depth, noisy_d, torch.full_like(noisy_d, 999.0))
            dilated_fg_depth = -F.max_pool2d(-fg_depth, kernel_size=k_halo, stride=1, padding=pad_halo)

            # Add smooth edge noise and depth recess (+offset pushes into warm/red TURBO depth)
            halo_noise = _generate_correlated_noise(noisy_d.shape, self.edge_flicker_cluster_size, device)
            halo_jitter = halo_noise * 0.010 + self.edge_flicker_depth_offset
            halo_depth = torch.clamp(dilated_fg_depth + halo_jitter, min=self.min_depth, max=self.max_depth)

            # Assign flickering phantom depth to perimeter and inter-finger pixels
            noisy_d = torch.where(flicker_active, halo_depth, noisy_d)

        # ----------------------------------------------------------------------
        # 5. RealSense 2D Edge-Preserving Spatial Filter (Technique 4)
        # ----------------------------------------------------------------------
        # Simulates the RealSense SDK onboard IIR spatial filter:
        # Smooths surface noise and edge flickering into continuous, organic surfaces
        # while preserving sharp boundaries across large depth jumps (> delta_threshold).
        if self.enable_spatial_filter:
            noisy_d = self.apply_spatial_filter(
                noisy_d,
                radius=self.spatial_filter_radius,
                delta_threshold=self.spatial_delta_threshold,
                min_depth=self.min_depth,
                max_depth=self.max_depth,
                enable_hole_filling=self.enable_spatial_hole_filling,
                hole_fill_radius=self.spatial_hole_fill_radius,
                hole_fill_depth_offset=self.spatial_hole_fill_offset,
            )

        # ======================================================================
        # [COMMENTED OUT FOR STEP-BY-STEP TESTING]
        # ======================================================================
        # # 6. Specular Glare Saturation Dropout
        # if self.enable_specular_dropout:
        #     specular_dropout = (gray > 0.96) & (grad_mag < 0.02)
        #
        # # 7. Disparity Calculation & Sub-Pixel Discretization
        # safe_d = torch.clamp(d, min=0.01)
        # disp = fb / safe_d
        # if self.subpixel_levels > 0:
        #     disp_quant = torch.round(disp * self.subpixel_levels) / self.subpixel_levels
        #     noisy_d = fb / torch.clamp(disp_quant, min=1e-3)
        # ======================================================================

        res = noisy_d.view(orig_shape)
        return res

    @staticmethod
    def apply_spatial_filter(
        depth: torch.Tensor,
        radius: int = 4,
        delta_threshold: float = 0.040,
        min_depth: float = 0.07,
        max_depth: float = 0.50,
        enable_hole_filling: bool = True,
        hole_fill_radius: int = 14,
        hole_fill_depth_offset: float = 0.030,
    ) -> torch.Tensor:
        """Applies RealSense-style 2D separable edge-preserving spatial filter with inter-finger hole filling.

        Smooths depth within surfaces, softens transitions, and fills narrow inter-finger
        gaps with a warm/red recessed depth surface uniting the fingers.
        """
        valid = (depth >= min_depth) & (depth <= max_depth)
        if not valid.any():
            return depth

        orig_ndim = depth.ndim
        if depth.ndim == 2:
            d_4d = depth.unsqueeze(0).unsqueeze(0)
            valid_4d = valid.unsqueeze(0).unsqueeze(0)
        elif depth.ndim == 3:
            d_4d = depth.unsqueeze(1)
            valid_4d = valid.unsqueeze(1)
        else:
            d_4d = depth
            valid_4d = valid

        b, c, img_h, img_w = d_4d.shape
        sigma_s = float(radius)
        sigma_r = delta_threshold

        working_d = d_4d.clone()
        working_valid = valid_4d.clone()

        # ----------------------------------------------------------------------
        # A. Inter-Finger Morphological Gap Bridging & Hole Filling
        # ----------------------------------------------------------------------
        # Simulates RealSense SDK spatial filter hole-filling between adjacent fingers:
        # Closes narrow gaps between fingers and fills them with slightly deeper depth (+offset),
        # rendering them in warm orange/red in the colormap uniting the fingers.
        if enable_hole_filling and hole_fill_radius > 0:
            k_fill = max(3, hole_fill_radius * 2 + 1)
            pad_fill = k_fill // 2

            # Morphological Closing: Dilation followed by Erosion
            dilated = F.max_pool2d(valid_4d.float(), kernel_size=k_fill, stride=1, padding=pad_fill) > 0.5
            eroded = -F.max_pool2d(-dilated.float(), kernel_size=k_fill, stride=1, padding=pad_fill) > 0.5
            gap_mask = eroded & (~valid_4d)

            # Sample nearest foreground finger depth
            fg_d = torch.where(valid_4d, d_4d, torch.full_like(d_4d, 999.0))
            dilated_fg_d = -F.max_pool2d(-fg_d, kernel_size=k_fill, stride=1, padding=pad_fill)

            # Recess bridged depth by +offset so it appears warm/red in Turbo colormap
            bridged_d = torch.clamp(dilated_fg_d + hole_fill_depth_offset, min=min_depth, max=max_depth)

            # Inject bridged surface into the inter-finger gaps
            working_d = torch.where(gap_mask, bridged_d, working_d)
            working_valid = working_valid | gap_mask

        # ----------------------------------------------------------------------
        # B. RealSense Separable Edge-Preserving Smoothing Pass
        # ----------------------------------------------------------------------
        # 1. Horizontal Pass
        weights_sum_h = torch.zeros_like(working_d)
        depth_sum_h = torch.zeros_like(working_d)
        pad_h = F.pad(working_d, (radius, radius, 0, 0), mode="replicate")
        pad_v_h = F.pad(working_valid.float(), (radius, radius, 0, 0), mode="constant", value=0.0)

        for dx in range(-radius, radius + 1):
            d_shifted = pad_h[:, :, :, (radius + dx) : (radius + dx + img_w)]
            v_shifted = pad_v_h[:, :, :, (radius + dx) : (radius + dx + img_w)]
            diff = torch.abs(d_shifted - working_d)
            w_spatial = math.exp(-(dx ** 2) / (2.0 * sigma_s ** 2))
            w_range = torch.exp(-(diff ** 2) / (2.0 * sigma_r ** 2))
            wt = w_spatial * w_range * v_shifted

            depth_sum_h += d_shifted * wt
            weights_sum_h += wt

        d_h = torch.where(weights_sum_h > 1e-4, depth_sum_h / weights_sum_h, working_d)

        # 2. Vertical Pass
        weights_sum_v = torch.zeros_like(d_h)
        depth_sum_v = torch.zeros_like(d_h)
        valid_h = (d_h >= min_depth) & (d_h <= max_depth)
        pad_v = F.pad(d_h, (0, 0, radius, radius), mode="replicate")
        pad_v_v = F.pad(valid_h.float(), (0, 0, radius, radius), mode="constant", value=0.0)

        for dy in range(-radius, radius + 1):
            d_shifted = pad_v[:, :, (radius + dy) : (radius + dy + img_h), :]
            v_shifted = pad_v_v[:, :, (radius + dy) : (radius + dy + img_h), :]
            diff = torch.abs(d_shifted - d_h)
            w_spatial = math.exp(-(dy ** 2) / (2.0 * sigma_s ** 2))
            w_range = torch.exp(-(diff ** 2) / (2.0 * sigma_r ** 2))
            wt = w_spatial * w_range * v_shifted

            depth_sum_v += d_shifted * wt
            weights_sum_v += wt

        filtered = torch.where(weights_sum_v > 1e-4, depth_sum_v / weights_sum_v, working_d)
        filtered = torch.where(working_valid, filtered, torch.zeros_like(filtered))

        if orig_ndim == 2:
            return filtered.squeeze(0).squeeze(0)
        elif orig_ndim == 3:
            return filtered.squeeze(1)
        return filtered


def apply_realsense_d405_noise(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    enable_low_contrast: bool | None = None,
    enable_voronoi_dropout: bool | None = None,
    enable_edge_flicker: bool | None = None,
    enable_spatial_filter: bool | None = None,
    enable_stereo_occlusion: bool | None = None,
    occlusion_side: str | None = None,
    contrast_threshold: float | None = None,
    depth_noise_std: float | None = None,
    edge_flicker_prob: float | None = None,
    edge_flicker_cluster_size: int | None = None,
    edge_halo_radius: int | None = None,
    edge_flicker_depth_offset: float | None = None,
    spatial_filter_radius: int | None = None,
    spatial_delta_threshold: float | None = None,
    enable_spatial_hole_filling: bool | None = None,
    spatial_hole_fill_radius: int | None = None,
    spatial_hole_fill_offset: float | None = None,
    voronoi_cell_size: int = 20,
    voronoi_spot_radius: float = 7.0,
    voronoi_density: float = 0.60,
    voronoi_contrast_threshold: float = 0.008,
    min_depth: float = 0.07,
    max_depth: float = 0.30,
) -> torch.Tensor:
    """Convenience functional interface to apply realistic RealSense D405 noise and dropouts."""
    model = RealSenseD405NoiseModel(
        min_depth=min_depth,
        max_depth=max_depth,
        enable_low_contrast=enable_low_contrast,
        enable_voronoi_dropout=enable_voronoi_dropout,
        enable_edge_flicker=enable_edge_flicker,
        enable_spatial_filter=enable_spatial_filter,
        enable_stereo_occlusion=enable_stereo_occlusion,
        occlusion_side=occlusion_side,
        contrast_threshold=contrast_threshold,
        depth_noise_std=depth_noise_std,
        edge_flicker_prob=edge_flicker_prob,
        edge_flicker_cluster_size=edge_flicker_cluster_size,
        edge_halo_radius=edge_halo_radius,
        edge_flicker_depth_offset=edge_flicker_depth_offset,
        spatial_filter_radius=spatial_filter_radius,
        spatial_delta_threshold=spatial_delta_threshold,
        enable_spatial_hole_filling=enable_spatial_hole_filling,
        spatial_hole_fill_radius=spatial_hole_fill_radius,
        spatial_hole_fill_offset=spatial_hole_fill_offset,
        voronoi_cell_size=voronoi_cell_size,
        voronoi_spot_radius=voronoi_spot_radius,
        voronoi_density=voronoi_density,
        voronoi_contrast_threshold=voronoi_contrast_threshold,
    )
    return model(rgb, depth)


def render_realsense_d405_dashboard(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    window_name: str | None = "Intel RealSense D405 (Wrist Camera) - RGB & Realistic Depth",
    apply_noise: bool = True,
    enable_low_contrast: bool | None = None,
    contrast_threshold: float | None = None,
    enable_voronoi_dropout: bool | None = None,
    enable_edge_flicker: bool | None = None,
    enable_spatial_filter: bool | None = None,
    enable_stereo_occlusion: bool | None = None,
    occlusion_side: str | None = None,
    view_width: int = 640,
    view_height: int = 360,
) -> np.ndarray:
    """Renders a real-time side-by-side OpenCV dashboard showing D405 RGB and Realistic Depth.

    Args:
        rgb: Rendered RGB tensor (..., H, W, 3/4)
        depth: Distance-to-image-plane tensor (..., H, W)
        window_name: OpenCV window title (or None to skip calling cv2.imshow)
        apply_noise: If True, applies the realistic RealSenseD405NoiseModel before visualization.
        enable_low_contrast: Flag to toggle Step 2 full low-contrast dropout filter (None uses DEFAULT_ENABLE_LOW_CONTRAST).
        contrast_threshold: Sobel gradient threshold for low-contrast dropout (None uses DEFAULT_CONTRAST_THRESHOLD).
        enable_voronoi_dropout: Flag to toggle Step 2b Voronoi cellular dropout filter (None uses DEFAULT_ENABLE_VORONOI_DROPOUT).
        enable_edge_flicker: Flag to toggle Step 4 surface depth noise and edge flickering (None uses DEFAULT_ENABLE_EDGE_FLICKER).
        enable_spatial_filter: Flag to toggle Step 5 RealSense edge-preserving spatial filter (None uses DEFAULT_ENABLE_SPATIAL_FILTER).
        enable_stereo_occlusion: Flag to toggle Step 3 stereo parallax occlusion shadows (None uses DEFAULT_ENABLE_STEREO_OCCLUSION).
        occlusion_side: Occlusion shadow side ("left", "right", or "both"). None uses DEFAULT_OCCLUSION_SIDE ("left").
        view_width: Display width of each individual feed in pixels (default 640).
        view_height: Display height of each individual feed in pixels (default 360).

    Returns:
        Composed BGR dashboard image as a numpy uint8 array of shape (view_height + 38, view_width * 2, 3).
    """
    # 1. Apply physical D405 noise & dropout if requested
    if apply_noise:
        noisy_d = apply_realsense_d405_noise(
            rgb,
            depth,
            enable_low_contrast=enable_low_contrast,
            contrast_threshold=contrast_threshold,
            enable_voronoi_dropout=enable_voronoi_dropout,
            enable_edge_flicker=enable_edge_flicker,
            enable_spatial_filter=enable_spatial_filter,
            enable_stereo_occlusion=enable_stereo_occlusion,
            occlusion_side=occlusion_side,
        )
    else:
        noisy_d = depth

    # 2. Extract single environment frame and convert to numpy
    if rgb.ndim == 4:
        rgb = rgb[0]
    if noisy_d.ndim == 4:
        noisy_d = noisy_d[0]
    if noisy_d.ndim == 3 and noisy_d.shape[-1] == 1:
        noisy_d = noisy_d.squeeze(-1)

    rgb_np = rgb.detach().cpu().numpy()
    if rgb_np.max() <= 1.0 and rgb_np.dtype != np.uint8:
        rgb_u8 = (rgb_np * 255.0).astype(np.uint8)
    else:
        rgb_u8 = rgb_np.astype(np.uint8)

    if rgb_u8.shape[-1] == 4:
        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGBA2BGR)
    else:
        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)

    depth_np = noisy_d.detach().cpu().numpy().astype(np.float32)

    # 3. Stereo Lock percentage (pixels with valid depth retained within D405 optical range 0.07m - 0.50m)
    valid_mask = (depth_np >= 0.07) & (depth_np <= 0.50)
    lock_pct = float(valid_mask.mean()) * 100.0

    # 4. Realistic RealSense Colormapping
    # Near (0.07m) = Cold/Blue, Far (0.30m) = Warm/Red
    d_clamped = np.clip(depth_np, 0.07, 0.30)
    norm_d = (d_clamped - 0.07) / (0.30 - 0.07)
    u8 = (norm_d * 255.0).astype(np.uint8)
    depth_col = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    # Out-of-range (>50cm or <7cm) and low-contrast dropped pixels are colored pure black
    depth_col[~valid_mask] = (0, 0, 0)

    # 5. Resize feeds to target display resolution (preserving 16:9 aspect ratio)
    bgr_resized = cv2.resize(bgr, (view_width, view_height), interpolation=cv2.INTER_LINEAR)
    depth_resized = cv2.resize(depth_col, (view_width, view_height), interpolation=cv2.INTER_NEAREST)

    # 6. Top Header Banner (38px height)
    hdr_h = 38
    total_w = view_width * 2
    header = np.zeros((hdr_h, total_w, 3), dtype=np.uint8)
    header[:] = (35, 35, 35)

    eff_flicker = (
        RealSenseD405NoiseModel.DEFAULT_ENABLE_EDGE_FLICKER
        if enable_edge_flicker is None
        else enable_edge_flicker
    )
    eff_spatial = (
        RealSenseD405NoiseModel.DEFAULT_ENABLE_SPATIAL_FILTER
        if enable_spatial_filter is None
        else enable_spatial_filter
    )
    eff_voronoi = (
        RealSenseD405NoiseModel.DEFAULT_ENABLE_VORONOI_DROPOUT
        if enable_voronoi_dropout is None
        else enable_voronoi_dropout
    )
    eff_contrast = (
        RealSenseD405NoiseModel.DEFAULT_ENABLE_LOW_CONTRAST
        if enable_low_contrast is None
        else enable_low_contrast
    )
    eff_contrast_th = (
        RealSenseD405NoiseModel.DEFAULT_CONTRAST_THRESHOLD
        if contrast_threshold is None
        else contrast_threshold
    )
    eff_occlusion = (
        RealSenseD405NoiseModel.DEFAULT_ENABLE_STEREO_OCCLUSION
        if enable_stereo_occlusion is None
        else enable_stereo_occlusion
    )
    contrast_tag = f"ON(th={eff_contrast_th:g})" if eff_contrast else "OFF"
    status_str = f"Noise: {'ON' if eff_flicker else 'OFF'} | Spat: {'ON' if eff_spatial else 'OFF'} | Voronoi: {'ON' if eff_voronoi else 'OFF'} | Contrast: {contrast_tag} | Occl: {'ON' if eff_occlusion else 'OFF'}"

    UI_FONT = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(header, "Intel RealSense D405: Left Imager RGB (1280x720)", (14, 25), UI_FONT, 0.44, (220, 220, 220), 1)
    cv2.putText(header, f"RealSense D405: Depth (7-30cm) | {status_str} | Lock: {lock_pct:4.1f}%", (view_width + 14, 25), UI_FONT, 0.40, (0, 255, 255), 1)

    # 7. Assemble side-by-side dashboard
    content = np.hstack([bgr_resized, depth_resized])
    dashboard = np.vstack([header, content])

    # Separator lines
    cv2.line(dashboard, (view_width, 0), (view_width, view_height + hdr_h), (80, 80, 80), 2)
    cv2.line(dashboard, (0, hdr_h), (total_w, hdr_h), (80, 80, 80), 1)

    # 8. Show OpenCV window if title provided
    if window_name:
        try:
            cv2.imshow(window_name, dashboard)
            cv2.waitKey(1)
        except Exception:
            pass

    return dashboard


