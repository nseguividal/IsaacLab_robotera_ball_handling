# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive visualization script to preview the Robotera hand, balls, and RealSense cameras.

Usage:
    # Run with Isaac Lab python launcher
    python scripts/visualize_hand_cameras.py --enable_cameras
"""

import argparse

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
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg

# Import our robot and camera configurations
from robotera_ball_handling.robots import ROBOTERA_XHAND_RIGHT_CFG
from robotera_ball_handling.sensors import (
    FINGERTIP_CONTACT_SENSOR_CFG,
    REALSENSE_D405_WRIST_CFG,
    REALSENSE_D435_OVERVIEW_CFG,
    VISUO_TACTILE_FINGERTIPS_CFG,
    compute_visuo_tactile_contact_features,
)


# ==============================================================================
# Scene Configuration
# ==============================================================================
class HandCameraPreviewSceneCfg(InteractiveSceneCfg):
    """Scene for previewing the Robotera hand with cameras and dual balls."""

    # Ground Plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.95, 0.95, 0.95),
        ),
    )

    # Robotera Right Hand with cupped/curled finger initialization
    robot = ROBOTERA_XHAND_RIGHT_CFG.replace(
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

    # Ball 1 (Red manipulation sphere - cradled inside palm)
    ball1 = RigidObjectCfg(
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

    # Ball 2 (Blue manipulation sphere - cradled inside palm)
    ball2 = RigidObjectCfg(
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

    # RealSense D405 (Wrist-Mounted Macro RGB-D Camera)
    wrist_camera: TiledCameraCfg = REALSENSE_D405_WRIST_CFG

    # RealSense D435 (Static Overview RGB-D Camera)
    overview_camera: TiledCameraCfg = REALSENSE_D435_OVERVIEW_CFG

    # Visuo-Tactile Optical Sensor Array (5 Fingertip Gel Pads)
    tactile_cameras: TiledCameraCfg = VISUO_TACTILE_FINGERTIPS_CFG

    # Fingertip Physics Contact Sensor
    contact_forces: ContactSensorCfg = FINGERTIP_CONTACT_SENSOR_CFG


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Main visualization loop."""
    sim_dt = sim.get_physics_dt()
    count = 0

    print("\n" + "=" * 70)
    print(" [INFO] Robotera Hand, RealSense Cameras & Visuo-Tactile Sensors Preview Running!")
    print(" - Orbit with Right Mouse / Left Mouse in Omniverse GUI viewport")
    print(" - Press 'Space' in Isaac Sim to Play/Pause physics")
    print(" - Close the Isaac Sim window or press Ctrl+C in terminal to exit")
    print("=" * 70 + "\n")

    while simulation_app.is_running():
        # Step simulation physics & rendering
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        count += 1
        if count % 100 == 0:
            wrist_rgb = scene["wrist_camera"].data.output["rgb"]
            overview_rgb = scene["overview_camera"].data.output["rgb"]
            tactile_depth = scene["tactile_cameras"].data.output["distance_to_image_plane"]

            # Physics contact forces
            physics_contact = torch.norm(scene["contact_forces"].data.net_forces_w, dim=-1)

            # Convert visuo-tactile depth to 5-fingertip force array calibrated with contact
            force_array, total_forces = compute_visuo_tactile_contact_features(
                tactile_depth, contact_forces=physics_contact
            )
            f_avg = total_forces.mean().item()
            f_max = total_forces.max().item()
            f_min = total_forces.min().item()

            print(
                f"[Step {count:05d}] Wrist: {wrist_rgb.shape} | "
                f"Overview: {overview_rgb.shape} | "
                f"Tactile Forces -> Avg: {f_avg:.3f} N | Max: {f_max:.3f} N | Min: {f_min:.3f} N"
            )


def main():
    """Setup and run the preview scene."""
    sim_cfg = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.1,
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
    main()
    simulation_app.close()

