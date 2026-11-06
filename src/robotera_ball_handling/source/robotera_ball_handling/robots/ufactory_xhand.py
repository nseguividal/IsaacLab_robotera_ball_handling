# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the UFactory uf850 arm combined with the Robotera XHand dexterous hand."""

import os
from pxr import Gf, Sdf, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.utils import clone
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .robotera_xhand import (
    ROBOTERA_XHAND_DIR,
    ROBOTERA_XHAND_RIGHT_URDF,
    ROBOTERA_RIGHT_FINGERTIP_NAMES,
    ROBOTERA_RIGHT_WRIST_LINK,
    ROBOTERA_RIGHT_PALM_LINK,
)

# Wristless Robotera XHand URDF (palm mounted directly to arm flange)
ROBOTERA_XHAND_RIGHT_WRISTLESS_URDF = os.path.join(
    ROBOTERA_XHAND_DIR, "urdf", "robotera_xhand_right_wristless.urdf"
)

# UFactory uf850 USD Asset Path in Isaac Sim / Nucleus
UFACTORY_UF850_USD = f"{ISAAC_NUCLEUS_DIR}/Robots/Ufactory/uf850/uf850.usd"


@clone
def spawn_ufactory_uf850_xhand(
    prim_path: str,
    cfg: sim_utils.SpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawns the UFactory uf850 arm and rigidly attaches the Robotera XHand palm directly to the tool flange (link6).

    Resulting Articulation:
    - Base: link_base (bolted to table at translation)
    - Arm: 6 revolute joints (joint1 to joint6)
    - Flange joint: fixed joint connecting link6 directly to right_hand_base_link (palm mount plate)
    - Hand: 12 revolute finger joints (Thumb, Index, Middle, Ring, Pinky)
    - Total: 18 active DoFs in a single unified PhysX articulation tree (no wrist joints).
    """
    # 1. Spawn UFactory uf850 arm from USD
    arm_usd_cfg = sim_utils.UsdFileCfg(
        usd_path=UFACTORY_UF850_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
        ),
    )
    arm_prim = sim_utils.spawn_from_usd(
        prim_path, arm_usd_cfg, translation=translation, orientation=orientation, **kwargs
    )
    stage = arm_prim.GetStage()
    resolved_arm_path = str(arm_prim.GetPath())

    # 2. Locate arm flange link in the arm USD before spawning the hand
    flange_prim = None
    flange_candidates = ["link6", "link_eef", "tool0", "flange", "link_tool"]
    for p in stage.Traverse():
        path_str = str(p.GetPath())
        if path_str.startswith(resolved_arm_path):
            if p.GetName() in flange_candidates:
                flange_prim = p
                break
    if flange_prim is None:
        for cand in flange_candidates:
            test_p = stage.GetPrimAtPath(f"{resolved_arm_path}/{cand}")
            if test_p.IsValid():
                flange_prim = test_p
                break
    if flange_prim is None:
        available_prims = [
            p.GetName() for p in stage.Traverse() if str(p.GetPath()).startswith(resolved_arm_path)
        ]
        raise RuntimeError(
            f"Could not locate flange link in UFactory uf850 arm. Available prims: {available_prims}"
        )
    flange_path = str(flange_prim.GetPath())
    print(f"[INFO] Located UFactory uf850 tool flange at: {flange_path}")

    # 3. Spawn Robotera XHand from URDF under resolved_arm_path/xhand (Wristless palm version)
    hand_prim_path = f"{resolved_arm_path}/xhand"
    hand_urdf_cfg = sim_utils.UrdfFileCfg(
        asset_path=ROBOTERA_XHAND_RIGHT_WRISTLESS_URDF,
        fix_base=False,  # Dynamic root, attached via FixedJoint to arm flange
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
    )
    # In the XHand URDF, the fingers extend along -Z from the palm base link.
    # To orient the palm pointing outward from the tool flange (+Z of link6),
    # we rotate 180° around the X axis (quat: w=0.0, x=1.0, y=0.0, z=0.0).
    hand_prim = sim_utils.spawn_from_urdf(
        hand_prim_path,
        hand_urdf_cfg,
        translation=(0.0, 0.0, 0.0),
        orientation=(0.0, 1.0, 0.0, 0.0),
    )
    resolved_hand_path = str(hand_prim.GetPath())

    # 4. Locate palm root mount link in the hand prim
    mount_link_prim = None
    mount_candidates = ["right_hand_base_link", "right_hand_link"]
    for p in stage.Traverse():
        path_str = str(p.GetPath())
        if path_str.startswith(resolved_hand_path):
            if p.GetName() in mount_candidates:
                mount_link_prim = p
                break
    if mount_link_prim is None:
        for cand in mount_candidates:
            test_p = stage.GetPrimAtPath(f"{resolved_hand_path}/{cand}")
            if test_p.IsValid():
                mount_link_prim = test_p
                break
    if mount_link_prim is None:
        raise RuntimeError(f"Could not locate palm mount link in Robotera XHand under: {resolved_hand_path}")
    mount_link_path = str(mount_link_prim.GetPath())
    print(f"[INFO] Located Robotera XHand palm mount link at: {mount_link_path}")

    # 5. Strip ArticulationRootAPI from the hand prims so the arm + hand forms a single articulation
    for p in stage.Traverse():
        path_str = str(p.GetPath())
        if path_str.startswith(resolved_hand_path):
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                print(f"[INFO] Removed UsdPhysics.ArticulationRootAPI from {path_str}")
            try:
                p.RemoveAPI("PhysicsArticulationRootAPI")
            except Exception:
                pass
            try:
                from pxr import PhysxSchema
                if p.HasAPI(PhysxSchema.PhysxArticulationRootAPI):
                    p.RemoveAPI(PhysxSchema.PhysxArticulationRootAPI)
                    print(f"[INFO] Removed PhysxSchema.PhysxArticulationRootAPI from {path_str}")
                p.RemoveAPI("PhysxArticulationRootAPI")
            except Exception:
                pass

    # 6. Create PhysX Fixed Joint binding arm flange directly to palm mount link
    joint_prim_path = f"{flange_path}/flange_to_palm_fixed_joint"
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(joint_prim_path))
    fixed_joint.CreateBody0Rel().SetTargets([Sdf.Path(flange_path)])
    fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(mount_link_path)])
    # Explicitly author local joint frames:
    # 180° rotation around X so hand fingers (-Z in URDF) point outward along the flange (+Z)
    fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(0.0, Gf.Vec3f(1.0, 0.0, 0.0)))
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    print(f"[INFO] Successfully rigidly bound arm flange ({flange_path}) directly to palm ({mount_link_path}).")

    return arm_prim


# ==============================================================================
# Complete Articulation Configuration: UFactory uf850 + Robotera XHand (20 DoFs)
# ==============================================================================

@sim_utils.configclass
class UfactoryXHandSpawnerCfg(sim_utils.SpawnerCfg):
    """Custom spawner config combining uf850 arm USD with Robotera XHand URDF."""
    func = spawn_ufactory_uf850_xhand


UFACTORY_UF850_XHAND_CFG = ArticulationCfg(
    spawn=UfactoryXHandSpawnerCfg(),
    init_state=ArticulationCfg.InitialStateCfg(
        # Position base firmly on top of table surface:
        # Table center: pos=(0.20, 0.0, 0.30), size=(0.80, 1.00, 0.04) -> Table top surface Z = 0.32m, X range [-0.20, +0.60]
        pos=(-0.10, 0.0, 0.32),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # UFactory uf850 6-DoF Arm: Natural ready-to-reach hover pose over table center
            "joint1": 0.0,    # Base yaw: facing forward along +X
            "joint2": -0.35,  # Shoulder pitch: leaning slightly forward
            "joint3": -0.70,  # Elbow pitch: bent forward (limits: [-4.224, 0.061])
            "joint4": 0.0,    # Forearm roll
            "joint5": -0.35,  # Wrist pitch: leveling hand with table
            "joint6": 0.0,    # Tool flange roll
            # Robotera XHand Fingers (12 DoFs): Gentle open-hand ready grasp (no wrist joints)
            "right_hand_thumb_bend_joint": 0.40,
            "right_hand_thumb_rota_joint1": 0.40,
            "right_hand_thumb_rota_joint2": 0.40,
            "right_hand_index_bend_joint": 0.0,
            "right_hand_index_joint1": 0.45,
            "right_hand_index_joint2": 0.50,
            "right_hand_mid_joint1": 0.45,
            "right_hand_mid_joint2": 0.50,
            "right_hand_ring_joint1": 0.45,
            "right_hand_ring_joint2": 0.50,
            "right_hand_pinky_joint1": 0.45,
            "right_hand_pinky_joint2": 0.50,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # UFactory uf850 Arm: High stiffness & damping for load bearing and trajectory tracking
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["^joint[1-6]$"],
            effort_limit=150.0,
            velocity_limit=3.14,
            stiffness=800.0,  # Kp
            damping=60.0,     # Kd
        ),
        # Robotera XHand Fingers: Compliant for soft contact grasping
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*right_hand_.*_joint.*"],
            effort_limit=1.5,
            velocity_limit=14.0,
            stiffness=15.0,   # Kp
            damping=0.5,      # Kd
        ),
    },
)
