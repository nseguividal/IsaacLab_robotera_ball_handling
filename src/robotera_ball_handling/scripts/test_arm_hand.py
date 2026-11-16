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

# Takes the long URDF link name and turns it into a short label (for UI)
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
    print(" - Press Ctrl+C to exit")
    print("=" * 80 + "\n")

    # Hardware Latency Queue (2 physics steps = ~33.3 ms bus latency)
    latency_steps = 2
    cmd_latency_queue = deque(maxlen=latency_steps)         #if queue is full it will feed the robot the command of 1 step ago (dropping anything older)

    # Viscoelastic Tactile values
    tactile_dynamics = TactileSignalDynamics(
        decay_ratio=0.08,               # stress relaxation strength --> so that the measured force sags slightly over time when object presses the object
        relaxation_time_s=2.0,          # how fast the sag happens (in sec)
        drift_rate=0.003,               #simulates drift so readings slowly deviate a bit over time
        noise_std=0.004,                # sensor jitter --> adds small px gaussian noise every step (in N)
    )

    while simulation_app.is_running():
        # Drive coordinated arm reach + finger grasp motion

        # We generate the motion targets (move with a oscillation from -1.0 tp 1.0 multiplied by a weight)
        t = step * 0.02
        alpha_arm = math.sin(t)
        alpha_hand = 0.50 * (1.0 - math.cos(t * 1.5))

        cur_targets = target_pos.clone()

        for j_idx, j_name in enumerate(robot.data.joint_names):
            # Arm Joint Control: gentle reaching oscillation over the table 
            # hardcoded arm joint angles
            if j_name == "joint1":
                cur_targets[:, j_idx] = 0.0 + alpha_arm * 0.20   # Yaw sway: +/- 0.2 rad
            elif j_name == "joint2":
                cur_targets[:, j_idx] = -0.35 + alpha_arm * 0.15 # Shoulder reach
            elif j_name == "joint3":
                cur_targets[:, j_idx] = -0.70 - alpha_arm * 0.15 # Elbow reach (limits: [-4.224, 0.061])
            elif j_name == "joint5":
                cur_targets[:, j_idx] = -0.35 - alpha_arm * 0.10 # Keep wrist level
            # Hand Finger Control: gentle grasp and release 
            # hardcoded finger angles
            elif "thumb_bend" in j_name:
                cur_targets[:, j_idx] = 0.40 + alpha_hand * 0.30
            elif "thumb_rota" in j_name:
                cur_targets[:, j_idx] = 0.40 + alpha_hand * 0.60
            elif "joint1" in j_name and "hand" in j_name:
                cur_targets[:, j_idx] = 0.45 + alpha_hand * 0.80
            elif "joint2" in j_name and "hand" in j_name:
                cur_targets[:, j_idx] = 0.50 + alpha_hand * 0.80

        # Apply hardware command latency
        cmd_latency_queue.append(cur_targets)           # each step we push new command to the back 
        delayed_targets = cmd_latency_queue[0]          # sending oldest command (from the front) to the robot
        robot.set_joint_position_target(delayed_targets)

        # Step physics & rendering --> advances simulation one step
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        step += 1

        # Process tactile sensors
        tactile_depth = scene["tactile_cameras"].data.output["distance_to_image_plane"]     #reads fingertip depth (rendering a depth image where each pixel is the distance from camera/fingertip sensor plane to object)
        physics_contact = torch.norm(scene["contact_forces"].data.net_forces_w, dim=-1)     # reads contact forces and we turn the Fx, Fy, Fz into a single force magnitude

        force_array, total_forces = compute_visuo_tactile_contact_features(
            tactile_depth,
            contact_forces=physics_contact,
            dynamics_model=tactile_dynamics,
            dt=sim_dt,
        )

        #print status every 60 steps (1s at 60Hz)
        if step % 60 == 0:
            wrist_info = ""
            if hasattr(args_cli, "enable_cameras") and args_cli.enable_cameras:
                wrist_rgb_t = scene["wrist_camera"].data.output.get("rgb", None)
                if wrist_rgb_t is not None:
                    wrist_info = f" | D405 Wrist Cam: {tuple(wrist_rgb_t.shape[1:])}"
            print(f"[Step {step:05d}] Arm Reach alpha: {alpha_arm:+.2f} | Max Finger Force: {total_forces.max().item():.2f} N{wrist_info}")

        # show RealSense D405 Wrist Camera (mounted to link6 tool flange)
        if hasattr(args_cli, "enable_cameras") and args_cli.enable_cameras:
            wrist_rgb_t = scene["wrist_camera"].data.output.get("rgb", None)
            if wrist_rgb_t is not None and len(wrist_rgb_t) > 0:
                wrist_rgb = wrist_rgb_t[0].cpu().numpy()
                if wrist_rgb.shape[-1] == 4:
                    wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGBA2BGR)
                else:
                    wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)

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
        dt=1.0 / 60.0,              # sets 60Hz physics (rendered every step)
        render_interval=1,
        device=args_cli.device if hasattr(args_cli, "device") and args_cli.device else "cuda:0",
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
