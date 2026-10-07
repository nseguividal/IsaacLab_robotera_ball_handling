# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Test script to verify visuo-tactile and physical contact force sensing with closed fingers.

Usage:
    python scripts/test_tactile_grasp.py --enable_cameras
"""

import argparse
import math

# 1. Isaac Lab App Launcher
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Robotera fingertip tactile force sensing on dual balls.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Simulation imports after app launch"""
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.sim import SimulationCfg

from robotera_ball_handling.robots import ROBOTERA_XHAND_RIGHT_CFG
from robotera_ball_handling.sensors import (
    FINGERTIP_CONTACT_SENSOR_CFG,
    REALSENSE_D405_WRIST_CFG,
    REALSENSE_D435_OVERVIEW_CFG,
    VISUO_TACTILE_FINGERTIPS_CFG,
    compute_visuo_tactile_contact_features,
)


# ==============================================================================
# Cradle Grasp Scene Configuration
# ==============================================================================
class TactileGraspTestSceneCfg(InteractiveSceneCfg):
    """Scene with pre-curled fingers forming a cradle holding the two balls."""

    # Ground Plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2500.0,
            color=(0.95, 0.95, 0.95),
        ),
    )

    # Robotera Right Hand with cupped/curled finger cradle and high-stiffness actuators
    robot = ROBOTERA_XHAND_RIGHT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ROBOTERA_XHAND_RIGHT_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.40),
            rot=(0.5, 0.5, -0.5, -0.5),
            joint_pos={
                # Wrist
                "right_wrist_pitch_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                # Thumb opposed towards center
                "right_hand_thumb_bend_joint": 0.55,
                "right_hand_thumb_rota_joint1": 0.65,
                "right_hand_thumb_rota_joint2": 0.65,
                # Fingers curled inwards creating an enclosure
                "right_hand_index_bend_joint": -0.05,
                "right_hand_index_joint1": 0.95,
                "right_hand_index_joint2": 1.00,
                "right_hand_mid_joint1": 0.95,
                "right_hand_mid_joint2": 1.00,
                "right_hand_ring_joint1": 0.95,
                "right_hand_ring_joint2": 1.00,
                "right_hand_pinky_joint1": 0.95,
                "right_hand_pinky_joint2": 1.00,
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

    # Ball 1 (Single Red manipulation sphere fixed statically in place)
    ball1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball1",
        spawn=sim_utils.SphereCfg(
            radius=0.024,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,  # Immovable static body - will not move or fall
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
            pos=(0.18, -0.02, 0.418),  # Static fixed position
        ),
    )

    # Multi-Modal Sensors
    wrist_camera: TiledCameraCfg = REALSENSE_D405_WRIST_CFG
    overview_camera: TiledCameraCfg = REALSENSE_D435_OVERVIEW_CFG
    tactile_cameras: TiledCameraCfg = VISUO_TACTILE_FINGERTIPS_CFG
    # Fingertip Physics Contact Sensor (Filtered strictly to single Ball 1)
    contact_forces: ContactSensorCfg = FINGERTIP_CONTACT_SENSOR_CFG.replace(
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball1"]
    )


def render_ascii_taxel_map(force_matrix_8x8: torch.Tensor, threshold: float = 0.01) -> str:
    """Renders an 8x8 force matrix using # for uncompressed background and * for active contact force."""
    lines = []
    has_contact = (force_matrix_8x8.max().item() > threshold)
    if not has_contact:
        for _ in range(8):
            lines.append("       # # # # # # # #")
        return "\n".join(lines) + "\n       (Pad in open air / no contact)"

    for r in range(8):
        row_str = "       "
        for c in range(8):
            val = force_matrix_8x8[r, c].item()
            if val > threshold:
                row_str += "* "
            else:
                row_str += "# "
        lines.append(row_str)
    return "\n".join(lines)


def get_clean_finger_name(body_name: str) -> str:
    """Extract clean finger name from USD link name."""
    b_low = body_name.lower()
    if "thumb" in b_low:
        return "Thumb"
    elif "index" in b_low:
        return "Index"
    elif "mid" in b_low:
        return "Middle"
    elif "ring" in b_low:
        return "Ring"
    elif "pinky" in b_low:
        return "Pinky"
    return body_name


def run_grasp_test(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Grasp test loop applying active squeeze and logging tactile outputs."""
    sim_dt = sim.get_physics_dt()
    robot: Articulation = scene["robot"]
    step = 0

    # Sensor body names in PhysX (alphabetical order in USD: Index, Middle, Pinky, Ring, Thumb)
    sensor_body_names = scene["contact_forces"].body_names
    finger_names = [get_clean_finger_name(b) for b in sensor_body_names]
    index_sensor_idx = finger_names.index("Index") if "Index" in finger_names else 0

    # Target closing position setpoint
    target_pos = robot.data.default_joint_pos.clone()

    print("\n" + "=" * 78)
    print(" [INFO] Tactile Grasp & Contact Verification Test Running (Static Fixed Ball)!")
    print(" - Single Red Ball is STATIC & IMMOVABLE (kinematic, zero gravity) at (X=0.18, Y=-0.02, Z=0.418).")
    print(" - Ball will NOT fall, move, or roll away.")
    print(" - Sensor Body Order mapped from PhysX:")
    for i, (b_name, d_name) in enumerate(zip(sensor_body_names, finger_names)):
        print(f"     [Sensor Index {i}] -> {d_name:6s} ('{b_name}')")
    print(f" - Displaying 8x8 Tactile Map on Index Finger Pad (Sensor Index {index_sensor_idx})")
    print(" - Press 'Space' in GUI to Pause/Play | Press Ctrl+C to exit")
    print("=" * 78 + "\n")

    while simulation_app.is_running():
        # Drive full deep squeeze trajectory (alpha oscillating between 0.2 and 1.0)
        alpha = 0.40 * math.sin(step * 0.03) + 0.60
        cur_targets = target_pos.clone()
        for j_idx in range(len(robot.data.joint_names)):
            j_name = robot.data.joint_names[j_idx]
            if "joint1" in j_name:
                cur_targets[:, j_idx] = 0.85 + alpha * (1.80 - 0.85)  # Proximal flexion up to 1.80 rad
            elif "joint2" in j_name:
                cur_targets[:, j_idx] = 0.90 + alpha * (1.80 - 0.90)  # Distal flexion up to 1.80 rad
            elif "thumb_rota" in j_name:
                cur_targets[:, j_idx] = 0.65 + alpha * (1.45 - 0.65)  # Thumb opposition up to 1.45 rad
            elif "thumb_bend" in j_name:
                cur_targets[:, j_idx] = 0.55 + alpha * (0.68 - 0.55)  # Thumb adduction up to 0.68 rad

        robot.set_joint_position_target(cur_targets)

        # Step physics & rendering
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        step += 1
        if step % 50 == 0:
            tactile_depth = scene["tactile_cameras"].data.output["distance_to_image_plane"]
            physics_contact = torch.norm(scene["contact_forces"].data.net_forces_w, dim=-1)

            # Convert depth & contact forces into (5, 8, 8) force array and total forces
            force_array, total_forces = compute_visuo_tactile_contact_features(
                tactile_depth, contact_forces=physics_contact
            )

            # Flatten to 1D finger forces for single env: (5,)
            t_forces = total_forces.view(-1)
            f_array = force_array.view(5, 8, 8)

            print(f"\n--- [Step {step:05d} | Squeeze Intensity: {alpha*100:.1f}%] ---")
            p_forces = physics_contact.view(-1)
            for i, name in enumerate(finger_names):
                f_val = t_forces[i].item()
                p_val = p_forces[i].item()
                state = f"🟢 IN CONTACT (Force: {f_val:.3f} N | PhysX: {p_val:.3f} N)" if (f_val > 0.01 or p_val > 0.01) else "⚪ In Air (0.000 N)"
                print(f"   [Index {i}] {name:6s}: {state}")

            # Always display the Index Finger 8x8 tactile pressure map
            print(f"\n   [Fingertip 8x8 Tactile Pressure Map - Index Finger Pad (* = Force, # = Free)]:")
            print(render_ascii_taxel_map(f_array[index_sensor_idx], threshold=0.01))
            print("-" * 78)


def main():
    sim_cfg = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=1.5,
            restitution=0.0,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.35, 0.28, 0.58], target=[0.08, 0.0, 0.42])

    scene_cfg = TactileGraspTestSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()

    run_grasp_test(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()

