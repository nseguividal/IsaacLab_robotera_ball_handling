# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Camera sensor configurations modeled after Intel RealSense cameras."""

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg


# ==============================================================================
# Intel RealSense D405 - Wrist-Mounted Sub-Millimeter Macro Camera
# ==============================================================================
# The D405 is Intel's sub-millimeter close-range RGB-D camera designed for
# robotic wrists and high-precision dexterous in-hand manipulation.
# - Operating range: 7 cm to 50 cm (min depth: 0.05m, max depth: 0.6m)
# - Field of view: ~87° x 58°
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
        horizontal_aperture=3.6,
        vertical_aperture=2.4,
        clipping_range=(0.05, 0.60),  # D405 macro depth range (5cm to 60cm)
    ),
    width=128,
    height=128,
)


# ==============================================================================
# Intel RealSense D435 - Overview / Tabletop Camera
# ==============================================================================
# The D435 is a standard wide-field RGB-D camera providing global scene context.
# - Operating range: 0.2 m to 3.0 m
# - Field of view: ~86° x 57°
# - Placed statically overlooking the hand and manipulation arena
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
        horizontal_aperture=3.6,
        vertical_aperture=2.4,
        clipping_range=(0.15, 3.00),  # D435 standard range (15cm to 3m)
    ),
    width=128,
    height=128,
)

