# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Script Purpose: Diagnostic test to articulate each finger individually to verify joint mappings, visual kinematics, and tactile tensor index alignments.

Usage:
    python scripts/test_individual_fingers.py --enable_cameras
"""

import argparse
import math
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test individual finger motion and sensor alignment.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Simulation imports after app launch"""
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


class IndividualFingerSceneCfg(InteractiveSceneCfg):
    """Diagnostic scene with hand and a fixed red target sphere."""

    # Ground Plane & Lighting
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)),
    )

    # Robotera Right Hand
    robot = ROBOTERA_XHAND_RIGHT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={
            "wrist": ROBOTERA_XHAND_RIGHT_CFG.actuators["wrist"],
            "fingers": ROBOTERA_XHAND_RIGHT_CFG.actuators["fingers"].replace(
                stiffness=80.0,
                damping=2.0,
                effort_limit=10.0,
            ),
        },
    )

    # Static Target Ball placed at (0.18, -0.02, 0.418)
    ball1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball1",
        spawn=sim_utils.SphereCfg(
            radius=0.024,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
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

    # Multi-Modal Sensors
    wrist_camera: TiledCameraCfg = REALSENSE_D405_WRIST_CFG
    overview_camera: TiledCameraCfg = REALSENSE_D435_OVERVIEW_CFG
    tactile_cameras: TiledCameraCfg = VISUO_TACTILE_FINGERTIPS_CFG
    contact_forces: ContactSensorCfg = FINGERTIP_CONTACT_SENSOR_CFG.replace(
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball1"]
    )


def get_clean_finger_name(body_name: str) -> str:
    """Extract human-readable finger name from USD prim body name."""
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


def render_ascii_taxel_map(force_matrix_8x8: torch.Tensor, threshold: float = 0.01) -> str:
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
            row_str += "* " if val > threshold else "# "
        lines.append(row_str)
    return "\n".join(lines)


def run_finger_diagnostic(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    robot: Articulation = scene["robot"]
    step = 0

    # Sensor body names registered in PhysX ContactSensor
    sensor_body_names = scene["contact_forces"].body_names
    finger_display_names = [get_clean_finger_name(b) for b in sensor_body_names]

    print("\n" + "=" * 80)
    print(" [INFO] Individual Finger Motion & Sensor Index Diagnostic Running!")
    print(" - PhysX Sensor Registered Body Order (Alphabetical USD order):")
    for i, (b_name, d_name) in enumerate(zip(sensor_body_names, finger_display_names)):
        print(f"     [Sensor Index {i}] -> {d_name:6s}  (Body: '{b_name}')")
    print("\n - Each finger will flex and extend individually for 3 seconds in sequence:")
    print("   1. Thumb  -> 2. Index  -> 3. Middle  -> 4. Ring  -> 5. Pinky")
    print(" - Watch the Isaac Sim GUI viewport to confirm which physical finger moves!")
    print("=" * 80 + "\n")

    # Joint lists per finger
    finger_joint_groups = [
        ("Thumb",  ["thumb_bend", "thumb_rota1", "thumb_rota2"]),
        ("Index",  ["index_joint1", "index_joint2"]),
        ("Middle", ["mid_joint1", "mid_joint2"]),
        ("Ring",   ["ring_joint1", "ring_joint2"]),
        ("Pinky",  ["pinky_joint1", "pinky_joint2"]),
    ]

    cycle_steps = int(3.0 / sim_dt)  # 3.0s per finger (180 steps)
    target_pos = robot.data.default_joint_pos.clone()

    while simulation_app.is_running():
        # Determine which finger is currently active
        total_groups = len(finger_joint_groups)
        active_finger_idx = (step // cycle_steps) % total_groups
        active_finger_name, active_keywords = finger_joint_groups[active_finger_idx]

        # Sinusoidal motion on active finger: flex between 0.3 rad and 1.8 rad
        flex_val = 0.75 * math.sin(step * 0.08) + 1.05

        cur_targets = target_pos.clone()
        # Reset all fingers to relaxed neutral
        for j_idx, j_name in enumerate(robot.data.joint_names):
            if "joint1" in j_name or "joint2" in j_name:
                cur_targets[:, j_idx] = 0.30
            elif "thumb" in j_name:
                cur_targets[:, j_idx] = 0.20

        # Flex ONLY the active finger
        for j_idx, j_name in enumerate(robot.data.joint_names):
            if any(kw in j_name for kw in active_keywords):
                if "thumb_bend" in j_name:
                    cur_targets[:, j_idx] = min(0.65, flex_val * 0.45)
                else:
                    cur_targets[:, j_idx] = min(1.85, flex_val)

        robot.set_joint_position_target(cur_targets)

        # Step physics & rendering
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        step += 1
        if step % 30 == 0:
            tactile_depth = scene["tactile_cameras"].data.output["distance_to_image_plane"]
            physics_contact = torch.norm(scene["contact_forces"].data.net_forces_w, dim=-1)

            force_array, total_forces = compute_visuo_tactile_contact_features(
                tactile_depth, contact_forces=physics_contact
            )

            t_forces = total_forces.view(-1)
            p_forces = physics_contact.view(-1)
            f_array = force_array.view(5, 8, 8)

            print(f"\n--- [Step {step:05d} | Actively Wiggling: >>> {active_finger_name.upper()} <<<] ---")
            for i, d_name in enumerate(finger_display_names):
                f_val = t_forces[i].item()
                p_val = p_forces[i].item()
                is_active = (d_name == active_finger_name)
                marker = "👉" if is_active else "  "
                state = f"🟢 IN CONTACT (Tactile: {f_val:.3f} N | Newton: {p_val:.3f} N)" if (f_val > 0.01 or p_val > 0.01) else "⚪ In Air (0.000 N)"
                print(f" {marker} [Sensor Index {i}] {d_name:6s}: {state}")

            # Show ASCII tactile map for the currently active finger's sensor index
            # Find sensor index for active finger
            active_sensor_idx = None
            for idx, name in enumerate(finger_display_names):
                if name == active_finger_name:
                    active_sensor_idx = idx
                    break

            if active_sensor_idx is not None:
                print(f"\n   [8x8 Tactile Map for Active Finger: {active_finger_name} (Sensor Index {active_sensor_idx})]:")
                print(render_ascii_taxel_map(f_array[active_sensor_idx], threshold=0.01))
            print("-" * 80)


def main():
    sim_cfg = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=1,
        physx=sim_utils.PhysxCfg(
            solver_type=1,  # TGS (Temporal Gauss-Seidel / Newton-Raphson nonlinear solver)
            solve_articulation_contact_last=True,  # Prioritize dynamic contact resolution for grasping
            min_position_iteration_count=8,
            max_position_iteration_count=32,
            min_velocity_iteration_count=4,
            max_velocity_iteration_count=16,
            enable_external_forces_every_iteration=True,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.005,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=1.5,
            restitution=0.0,
            compliant_contact_stiffness=2000.0,
            compliant_contact_damping=50.0,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.35, 0.28, 0.58], target=[0.08, 0.0, 0.42])

    scene_cfg = IndividualFingerSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()

    run_finger_diagnostic(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()

