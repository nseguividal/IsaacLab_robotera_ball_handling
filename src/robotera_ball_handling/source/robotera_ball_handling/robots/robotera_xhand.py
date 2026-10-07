# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Robotera XHand dexterous hand."""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Base directory for assets
ROBOTERA_XHAND_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "robots", "xhand")
ROBOTERA_XHAND_RIGHT_URDF = os.path.join(ROBOTERA_XHAND_DIR, "urdf", "robotera_xhand_right.urdf")
ROBOTERA_XHAND_LEFT_URDF = os.path.join(ROBOTERA_XHAND_DIR, "urdf", "robotera_xhand_left.urdf")

# Fingertip link names (for contact & tactile sensors)
ROBOTERA_RIGHT_FINGERTIP_NAMES = [
    "right_hand_thumb_rota_tip",
    "right_hand_index_rota_tip",
    "right_hand_mid_tip",
    "right_hand_ring_tip",
    "right_hand_pinky_tip",
]

ROBOTERA_LEFT_FINGERTIP_NAMES = [
    "left_hand_thumb_rota_tip",
    "left_hand_index_rota_tip",
    "left_hand_mid_tip",
    "left_hand_ring_tip",
    "left_hand_pinky_tip",
]

# Wrist mounting link names (for RealSense D405 wrist camera attachment)
ROBOTERA_RIGHT_WRIST_LINK = "right_wrist_roll_link"
ROBOTERA_LEFT_WRIST_LINK = "left_wrist_roll_link"
ROBOTERA_RIGHT_PALM_LINK = "right_hand_link"
ROBOTERA_LEFT_PALM_LINK = "left_hand_link"


ROBOTERA_XHAND_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=ROBOTERA_XHAND_RIGHT_URDF,
        fix_base=True,
        merge_fixed_joints=False,  # Keep tip links and sensor frames intact
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Position 40cm above ground
        pos=(0.0, 0.0, 0.4),
        # Horizontal flat hand with forearm along -X, fingers along +X, and palm facing straight UP (+Z)
        rot=(0.5, 0.5, -0.5, -0.5),
        joint_pos={
            # Wrist joints
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            # Thumb
            "right_hand_thumb_bend_joint": 0.50,
            "right_hand_thumb_rota_joint1": 0.55,
            "right_hand_thumb_rota_joint2": 0.60,
            # Index
            "right_hand_index_bend_joint": 0.0,
            "right_hand_index_joint1": 0.85,
            "right_hand_index_joint2": 0.90,
            # Middle
            "right_hand_mid_joint1": 0.85,
            "right_hand_mid_joint2": 0.90,
            # Ring
            "right_hand_ring_joint1": 0.85,
            "right_hand_ring_joint2": 0.90,
            # Pinky
            "right_hand_pinky_joint1": 0.85,
            "right_hand_pinky_joint2": 0.90,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["right_wrist_.*"],
            effort_limit=35.0,    
            velocity_limit=4.19,
            stiffness=50.0, #Kp
            damping=2.0,    #Kd
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["right_hand_.*_joint.*"],
            effort_limit=1.5,
            velocity_limit=14.0,
            stiffness=15.0,    #Kp
            damping=0.5,       #Kd
        ),
    },
)

ROBOTERA_XHAND_LEFT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=ROBOTERA_XHAND_LEFT_URDF,
        fix_base=True,
        merge_fixed_joints=False,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        rot=(0.5, 0.5, -0.5, -0.5),
        joint_pos={
            # Wrist joints
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            # Thumb
            "left_hand_thumb_bend_joint": 0.50,
            "left_hand_thumb_rota_joint1": 0.55,
            "left_hand_thumb_rota_joint2": 0.60,
            # Index
            "left_hand_index_bend_joint": 0.0,
            "left_hand_index_joint1": 0.85,
            "left_hand_index_joint2": 0.90,
            # Middle
            "left_hand_mid_joint1": 0.85,
            "left_hand_mid_joint2": 0.90,
            # Ring
            "left_hand_ring_joint1": 0.85,
            "left_hand_ring_joint2": 0.90,
            # Pinky
            "left_hand_pinky_joint1": 0.85,
            "left_hand_pinky_joint2": 0.90,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["left_wrist_.*"],
            effort_limit=35.0,
            velocity_limit=4.19,
            stiffness=50.0,
            damping=2.0,
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["left_hand_.*_joint.*"],
            effort_limit=1.5,
            velocity_limit=14.0,
            stiffness=15.0,
            damping=0.5,
        ),
    },
)

