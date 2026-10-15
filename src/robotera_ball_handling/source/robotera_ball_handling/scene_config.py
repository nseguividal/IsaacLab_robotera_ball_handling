# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Centralized Scene Configurations for Robotera XHand simulation environments.

Provides standardized, reusable scene configurations:
- Environment: Simple Room (ceiling lights, wood panels) + Oak wood table
- Dexterous Hand: Robotera XHand (Right) with customizable joint postures & actuators
- Sensors: RealSense D405 (wrist), RealSense D435 (overview), 4x Ceiling cameras, Visuo-tactile fingertip cameras
- Manipulation objects: Standardized red & cyan manipulation spheres

Usage in any test script or training workflow:
    from robotera_ball_handling.scene_config import (
        TactileGraspTestSceneCfg,
        HandCameraPreviewSceneCfg,
        RoboteraWorkspaceSceneCfg,
    )
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

from robotera_ball_handling.robots import ROBOTERA_XHAND_RIGHT_CFG
from robotera_ball_handling.sensors import (
    CEILING_CAMERA_FRONT_LEFT_CFG,
    CEILING_CAMERA_FRONT_RIGHT_CFG,
    CEILING_CAMERA_REAR_LEFT_CFG,
    CEILING_CAMERA_REAR_RIGHT_CFG,
    FINGERTIP_CONTACT_SENSOR_CFG,
    REALSENSE_D405_WRIST_CFG,
    REALSENSE_D435_OVERVIEW_CFG,
    VISUO_TACTILE_FINGERTIPS_CFG,
)


# ==============================================================================
# Reusable Base Environment Assets
# ==============================================================================

ROOM_CFG = AssetBaseCfg(
    prim_path="/World/Room",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room.usd",
    ),
)

TABLE_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Table",
    spawn=sim_utils.CuboidCfg(
        size=(0.80, 1.00, 0.04),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak.mdl",
            project_uvw=True,
            texture_scale=(1.5, 1.5),
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=0.8,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0.20, 0.0, 0.30),
    ),
)


# ==============================================================================
# Manipulation Objects
# ==============================================================================

# Ball 1 (Primary Red manipulation sphere - radius 24mm, mass 50g)
BALL1_STATIC_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball1",
    spawn=sim_utils.SphereCfg(
        radius=0.024,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,  # Immovable static target for contact verification
            max_depenetration_velocity=5.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.15, 0.15), roughness=0.8),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=1.5,
            restitution=0.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.18, -0.02, 0.418),
    ),
)

BALL1_DYNAMIC_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball1",
    spawn=sim_utils.SphereCfg(
        radius=0.024,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.15, 0.15), roughness=0.8),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=1.5,
            restitution=0.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.18, -0.02, 0.418),
    ),
)

BALL2_DYNAMIC_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball2",
    spawn=sim_utils.SphereCfg(
        radius=0.024,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.75, 0.95), roughness=0.8),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=1.5,
            restitution=0.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.18, 0.03, 0.418),
    ),
)

# Ball 1 & 2 for Cupped Palm Cradle Preview
BALL1_CRADLE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball1",
    spawn=sim_utils.SphereCfg(
        radius=0.024,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.15, 0.15), roughness=0.3),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.90,
            dynamic_friction=0.75,
            restitution=0.05,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.080, -0.022, 0.440),
    ),
)

BALL2_CRADLE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball2",
    spawn=sim_utils.SphereCfg(
        radius=0.024,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.40, 0.95), roughness=0.3),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.90,
            dynamic_friction=0.75,
            restitution=0.05,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.080, 0.022, 0.440),
    ),
)


# ==============================================================================
# Robot Hand Posture & Actuator Presets
# ==============================================================================

# 1. Flat hand initialization with high finger stiffness (for dynamic grasp tests)
ROBOT_TACTILE_GRASP_CFG = ROBOTERA_XHAND_RIGHT_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    init_state=ROBOTERA_XHAND_RIGHT_CFG.init_state.replace(
        pos=(0.0, 0.0, 0.40),
        rot=(0.5, 0.5, -0.5, -0.5),
        joint_pos={
            # Wrist
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            # Thumb straight and static (0.0 rad)
            "right_hand_thumb_bend_joint": 0.0,
            "right_hand_thumb_rota_joint1": 0.0,
            "right_hand_thumb_rota_joint2": 0.0,
            # Fingers - initialized completely flat (0.0 rad)
            "right_hand_index_bend_joint": 0.0,
            "right_hand_index_joint1": 0.0,
            "right_hand_index_joint2": 0.0,
            "right_hand_mid_joint1": 0.0,
            "right_hand_mid_joint2": 0.0,
            "right_hand_ring_joint1": 0.0,
            "right_hand_ring_joint2": 0.0,
            "right_hand_pinky_joint1": 0.0,
            "right_hand_pinky_joint2": 0.0,
        },
    ),
    actuators={
        "wrist": ROBOTERA_XHAND_RIGHT_CFG.actuators["wrist"],
        "fingers": ROBOTERA_XHAND_RIGHT_CFG.actuators["fingers"].replace(
            stiffness=80.0,    # High stiffness so fingers press firmly onto ball
            damping=2.0,
            effort_limit=10.0, # Strong torque capability
        ),
    },
)

# 2. Cupped/curled finger initialization (for visual camera verification)
ROBOT_CUPOVER_PREVIEW_CFG = ROBOTERA_XHAND_RIGHT_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    init_state=ROBOTERA_XHAND_RIGHT_CFG.init_state.replace(
        pos=(0.0, 0.0, 0.40),
        rot=(0.5, 0.5, -0.5, -0.5),
        joint_pos={
            # Wrist
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            # Thumb opposition
            "right_hand_thumb_bend_joint": 0.50,
            "right_hand_thumb_rota_joint1": 0.55,
            "right_hand_thumb_rota_joint2": 0.60,
            # Fingers curled inwards around balls
            "right_hand_index_bend_joint": 0.0,
            "right_hand_index_joint1": 0.85,
            "right_hand_index_joint2": 0.90,
            "right_hand_mid_joint1": 0.85,
            "right_hand_mid_joint2": 0.90,
            "right_hand_ring_joint1": 0.85,
            "right_hand_ring_joint2": 0.90,
            "right_hand_pinky_joint1": 0.85,
            "right_hand_pinky_joint2": 0.90,
        },
    ),
)


# ==============================================================================
# Standardized Interactive Scene Configurations
# ==============================================================================

@configclass
class RoboteraWorkspaceSceneCfg(InteractiveSceneCfg):
    """Complete Robotera workspace scene with room, table, hand, full sensor suite, and 4 ceiling cameras."""

    # Environment & Table
    room: AssetBaseCfg = ROOM_CFG
    table: AssetBaseCfg = TABLE_CFG

    # Dexterous Hand
    robot: ArticulationCfg = ROBOT_TACTILE_GRASP_CFG

    # Manipulation Targets
    ball1: RigidObjectCfg = BALL1_STATIC_CFG

    # Sensor Suite: Wrist, Tabletop, Tactile
    wrist_camera: TiledCameraCfg = REALSENSE_D405_WRIST_CFG
    overview_camera: TiledCameraCfg = REALSENSE_D435_OVERVIEW_CFG
    tactile_cameras: TiledCameraCfg = VISUO_TACTILE_FINGERTIPS_CFG
    contact_forces: ContactSensorCfg = FINGERTIP_CONTACT_SENSOR_CFG.replace(
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball1"]
    )

    # 4 Overhead Ceiling Cameras
    ceiling_cam_front_left: TiledCameraCfg = CEILING_CAMERA_FRONT_LEFT_CFG
    ceiling_cam_front_right: TiledCameraCfg = CEILING_CAMERA_FRONT_RIGHT_CFG
    ceiling_cam_rear_left: TiledCameraCfg = CEILING_CAMERA_REAR_LEFT_CFG
    ceiling_cam_rear_right: TiledCameraCfg = CEILING_CAMERA_REAR_RIGHT_CFG


@configclass
class TactileGraspTestSceneCfg(InteractiveSceneCfg):
    """Standardized scene specifically optimized for test_tactile_grasp.py."""

    room: AssetBaseCfg = ROOM_CFG
    table: AssetBaseCfg = TABLE_CFG
    robot: ArticulationCfg = ROBOT_TACTILE_GRASP_CFG
    ball1: RigidObjectCfg = BALL1_STATIC_CFG

    wrist_camera: TiledCameraCfg = REALSENSE_D405_WRIST_CFG
    overview_camera: TiledCameraCfg = REALSENSE_D435_OVERVIEW_CFG
    tactile_cameras: TiledCameraCfg = VISUO_TACTILE_FINGERTIPS_CFG
    contact_forces: ContactSensorCfg = FINGERTIP_CONTACT_SENSOR_CFG.replace(
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball1"]
    )

    # 4 Overhead Ceiling Cameras
    ceiling_cam_front_left: TiledCameraCfg = CEILING_CAMERA_FRONT_LEFT_CFG
    ceiling_cam_front_right: TiledCameraCfg = CEILING_CAMERA_FRONT_RIGHT_CFG
    ceiling_cam_rear_left: TiledCameraCfg = CEILING_CAMERA_REAR_LEFT_CFG
    ceiling_cam_rear_right: TiledCameraCfg = CEILING_CAMERA_REAR_RIGHT_CFG


@configclass
class HandCameraPreviewSceneCfg(InteractiveSceneCfg):
    """Standardized scene specifically optimized for visualize_hand_cameras.py."""

    room: AssetBaseCfg = ROOM_CFG
    table: AssetBaseCfg = TABLE_CFG
    robot: ArticulationCfg = ROBOT_CUPOVER_PREVIEW_CFG
    ball1: RigidObjectCfg = BALL1_CRADLE_CFG
    ball2: RigidObjectCfg = BALL2_CRADLE_CFG

    wrist_camera: TiledCameraCfg = REALSENSE_D405_WRIST_CFG
    overview_camera: TiledCameraCfg = REALSENSE_D435_OVERVIEW_CFG
    tactile_cameras: TiledCameraCfg = VISUO_TACTILE_FINGERTIPS_CFG
    contact_forces: ContactSensorCfg = FINGERTIP_CONTACT_SENSOR_CFG

    # 4 Overhead Ceiling Cameras
    ceiling_cam_front_left: TiledCameraCfg = CEILING_CAMERA_FRONT_LEFT_CFG
    ceiling_cam_front_right: TiledCameraCfg = CEILING_CAMERA_FRONT_RIGHT_CFG
    ceiling_cam_rear_left: TiledCameraCfg = CEILING_CAMERA_REAR_LEFT_CFG
    ceiling_cam_rear_right: TiledCameraCfg = CEILING_CAMERA_REAR_RIGHT_CFG
