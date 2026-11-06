# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Script Purpose: Interactive verification of UFactory uf850 arm mounted on table with Robotera XHand.

Usage:
    python src/robotera_ball_handling/scripts/test_arm_hand.py --enable_cameras
"""

import argparse
import math
import time
from collections import deque

# 1. Isaac Lab App Launcher
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test UFactory uf850 arm + Robotera XHand integration on table.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Simulation imports after app launch"""
import cv2
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg

from robotera_ball_handling.scene_config import UfactoryXHandSceneCfg
from robotera_ball_handling.sensors import (
    TactileSignalDynamics,
    apply_realsense_d405_noise,
    compute_visuo_tactile_contact_features,
)


def get_clean_finger_name(body_name: str) -> str:
    """Extract clean display name from body name."""
    b_lower = body_name.lower()
    if "thumb" in b_lower:
        return "Thumb"
    elif "index" in b_lower:
        return "Index"
    elif "mid" in b_lower:
        return "Middle"
    elif "ring" in b_lower:
        return "Ring"
    elif "pinky" in b_lower:
        return "Pinky"
    return body_name


def run_arm_hand_test(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    robot: Articulation = scene["robot"]
    step = 0

    sensor_body_names = scene["contact_forces"].body_names
    finger_names = [get_clean_finger_name(b) for b in sensor_body_names]

    # Target home positions for all joints
    target_pos = robot.data.default_joint_pos.clone()

    print("\n" + "=" * 80)
    print(" [INFO] UFactory uf850 Arm + Robotera XHand (Direct Palm Mount) Running!")
    print(f" - Base link mounted on table at: {robot.data.root_pos_w[0].cpu().numpy()}")
    print(f" - Unified Articulation Total DoFs: {len(robot.data.joint_names)} (6 Arm + 12 Finger DoFs, 0 Wrist)")
    for idx, j_name in enumerate(robot.data.joint_names):
        category = "ARM" if ("joint" in j_name and not "hand" in j_name) else "FINGER"
        print(f"     [{idx:02d}] {j_name:30s} -> [{category}]")
    print(" - Press 'Space' in GUI to Pause/Play | Press Ctrl+C to exit")
    print("=" * 80 + "\n")

    # Hardware Latency Queue (2 physics steps = ~33.3 ms bus latency)
    latency_steps = 2
    cmd_latency_queue = deque(maxlen=latency_steps)

    # Viscoelastic Tactile Dynamics
    tactile_dynamics = TactileSignalDynamics(
        decay_ratio=0.08,
        relaxation_time_s=2.0,
        drift_rate=0.003,
        noise_std=0.004,
    )

    while simulation_app.is_running():
        # Drive coordinated arm reach + finger grasp motion
        t = step * 0.02
        alpha_arm = math.sin(t)
        alpha_hand = 0.50 * (1.0 - math.cos(t * 1.5))

        cur_targets = target_pos.clone()

        for j_idx, j_name in enumerate(robot.data.joint_names):
            # Arm Joint Control: gentle reaching oscillation over the table
            if j_name == "joint1":
                cur_targets[:, j_idx] = 0.0 + alpha_arm * 0.20   # Yaw sway: +/- 0.2 rad
            elif j_name == "joint2":
                cur_targets[:, j_idx] = -0.35 + alpha_arm * 0.15 # Shoulder reach
            elif j_name == "joint3":
                cur_targets[:, j_idx] = -0.70 - alpha_arm * 0.15 # Elbow reach (limits: [-4.224, 0.061])
            elif j_name == "joint5":
                cur_targets[:, j_idx] = -0.35 - alpha_arm * 0.10 # Keep wrist level
            # Hand Finger Control: gentle grasp and release
            elif "thumb_bend" in j_name:
                cur_targets[:, j_idx] = 0.40 + alpha_hand * 0.30
            elif "thumb_rota" in j_name:
                cur_targets[:, j_idx] = 0.40 + alpha_hand * 0.60
            elif "joint1" in j_name and "hand" in j_name:
                cur_targets[:, j_idx] = 0.45 + alpha_hand * 0.80
            elif "joint2" in j_name and "hand" in j_name:
                cur_targets[:, j_idx] = 0.50 + alpha_hand * 0.80

        # Apply hardware command latency
        cmd_latency_queue.append(cur_targets)
        delayed_targets = cmd_latency_queue[0]
        robot.set_joint_position_target(delayed_targets)

        # Step physics & rendering
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        step += 1

        # Process tactile sensors
        tactile_depth = scene["tactile_cameras"].data.output["distance_to_image_plane"]
        physics_contact = torch.norm(scene["contact_forces"].data.net_forces_w, dim=-1)

        force_array, total_forces = compute_visuo_tactile_contact_features(
            tactile_depth,
            contact_forces=physics_contact,
            dynamics_model=tactile_dynamics,
            dt=sim_dt,
        )

        if step % 60 == 0:
            wrist_info = ""
            if hasattr(args_cli, "enable_cameras") and args_cli.enable_cameras:
                wrist_rgb_t = scene["wrist_camera"].data.output.get("rgb", None)
                if wrist_rgb_t is not None:
                    wrist_info = f" | D405 Wrist Cam: {tuple(wrist_rgb_t.shape[1:])}"
            print(f"[Step {step:05d}] Arm Reach alpha: {alpha_arm:+.2f} | Max Finger Force: {total_forces.max().item():.2f} N{wrist_info}")

        # RealSense D405 Wrist Camera (mounted to link6 tool flange)
        if hasattr(args_cli, "enable_cameras") and args_cli.enable_cameras:
            wrist_rgb_t = scene["wrist_camera"].data.output.get("rgb", None)
            if wrist_rgb_t is not None and len(wrist_rgb_t) > 0:
                wrist_rgb = wrist_rgb_t[0].cpu().numpy()
                if wrist_rgb.shape[-1] == 4:
                    wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGBA2BGR)
                else:
                    wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
                
                # Periodically save snapshot to disk
                if step % 300 == 0:
                    cv2.imwrite("/tmp/d405_wrist_snapshot.jpg", wrist_bgr)

                # Try showing OpenCV window (safely ignores if OpenCV is headless/no GUI backend)
                try:
                    preview = cv2.resize(wrist_bgr, (640, 360))
                    cv2.putText(
                        preview,
                        f"D405 Wrist Cam [Flange link6] - Step {step:05d}",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("RealSense D405 Wrist Camera (Mounted to Flange)", preview)
                    cv2.waitKey(1)
                except (cv2.error, Exception):
                    pass


def main():
    sim_cfg = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=1,
        device=args_cli.device if hasattr(args_cli, "device") and args_cli.device else "cuda:0",
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            solve_articulation_contact_last=True,
            min_position_iteration_count=8,
            max_position_iteration_count=32,
            min_velocity_iteration_count=4,
            max_velocity_iteration_count=16,
            enable_external_forces_every_iteration=True,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.005,
            gpu_max_rigid_contact_count=2**16,
            gpu_max_rigid_patch_count=2**13,
            gpu_found_lost_pairs_capacity=2**14,
            gpu_total_aggregate_pairs_capacity=2**13,
            gpu_max_soft_body_contacts=2**10,
            gpu_max_particle_contacts=2**10,
            gpu_heap_capacity=16 * 1024 * 1024,
            gpu_temp_buffer_capacity=4 * 1024 * 1024,
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
    # Camera looking at arm mounted on the table
    sim.set_camera_view(eye=[0.65, 0.75, 0.80], target=[0.05, 0.0, 0.38])

    scene = InteractiveScene(UfactoryXHandSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    run_arm_hand_test(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
