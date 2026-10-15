# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Script Purpose: Visual inspection of the Robotera hand kinematics, dual manipulation balls, and RealSense wrist/overview cameras in the viewport.

Usage:
    python scripts/visualize_hand_cameras.py --enable_cameras
"""

import argparse
import time

# 1. Isaac Lab App Launcher (Must be initialized before importing omni/pxr/torch modules)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize Robotera XHand with RealSense cameras and balls.")
# Add standard Isaac Lab app launcher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Ensure cameras are enabled in the simulator viewport
args_cli.enable_cameras = True

# Launch the Omniverse simulation application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Simulation imports after app launch"""
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, TiledCamera
from isaaclab.sim import SimulationCfg

# Import standardized scene configuration and camera utilities
from robotera_ball_handling.scene_config import HandCameraPreviewSceneCfg
from robotera_ball_handling.sensors import (
    apply_realsense_d405_noise,
    compute_visuo_tactile_contact_features,
    render_realsense_d405_dashboard,
)


# RealSense D405 Filter Flags are controlled directly in cameras.py (RealSenseD405NoiseModel.DEFAULT_ENABLE_*)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Main visualization loop."""
    sim_dt = sim.get_physics_dt()
    count = 0

    # Camera Specifications (Head, Wrist, Tactile)
    head_res = f"{scene['overview_camera'].cfg.width}x{scene['overview_camera'].cfg.height}"
    wrist_res = f"{scene['wrist_camera'].cfg.width}x{scene['wrist_camera'].cfg.height}"
    tactile_res = f"{scene['tactile_cameras'].cfg.width}x{scene['tactile_cameras'].cfg.height}"
    target_fps = 1.0 / (sim_dt * sim.cfg.render_interval)

    # Loop Frequencies & Periods (Policy vs Control)
    ctrl_dt = sim_dt
    ctrl_freq = 1.0 / ctrl_dt
    ctrl_period_ms = ctrl_dt * 1000.0
    decimation = 2  # Standard policy decimation for dexterous manipulation
    policy_dt = ctrl_dt * decimation
    policy_freq = 1.0 / policy_dt
    policy_period_ms = policy_dt * 1000.0

    print("\n" + "=" * 70)
    print(" [INFO] Robotera Hand, RealSense Cameras & Visuo-Tactile Sensors Preview Running!")
    print(" - Execution & Simulation Loops:")
    print(f"     * High-Level Policy Loop : {policy_freq:5.1f} Hz  (Period: {policy_period_ms:5.1f} ms | Decimation: {decimation})")
    print(f"     * Low-Level Control Loop : {ctrl_freq:5.1f} Hz  (Period: {ctrl_period_ms:5.1f} ms | Joint PD & PhysX)")
    print(" - Camera Specifications:")
    print(f"     * Head Camera    (Overview D435) : Resolution {head_res}  | Target {target_fps:.0f} FPS")
    print(f"     * Wrist Camera   (Macro D405)    : Resolution {wrist_res}  | Target {target_fps:.0f} FPS")
    print(f"     * Tactile Sensor (5 Fingertips)  : Resolution {tactile_res}    | Target {target_fps:.0f} FPS")
    print(" - Orbit with Right Mouse / Left Mouse in Omniverse GUI viewport")
    print(" - Press 'Space' in Isaac Sim to Play/Pause physics")
    print(" - Close the Isaac Sim window or press Ctrl+C in terminal to exit")
    print("=" * 70 + "\n")

    while simulation_app.is_running():
        # Step simulation physics & rendering
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        # Retrieve wrist camera RGB & Depth and render live D405 OpenCV HUD
        wrist_rgb = scene["wrist_camera"].data.output["rgb"]
        wrist_depth = scene["wrist_camera"].data.output["distance_to_image_plane"]
        render_realsense_d405_dashboard(wrist_rgb, wrist_depth)


def main():
    """Setup and run the preview scene."""
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
            static_friction=0.90,
            dynamic_friction=0.75,
            restitution=0.05,
            compliant_contact_stiffness=2000.0,
            compliant_contact_damping=50.0,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set camera view in the Omniverse viewport
    sim.set_camera_view(eye=[0.4, 0.35, 0.65], target=[0.05, 0.0, 0.40])

    scene_cfg = HandCameraPreviewSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Reset scene
    sim.reset()
    scene.reset()

    # Run visualization
    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
        simulation_app.close()

