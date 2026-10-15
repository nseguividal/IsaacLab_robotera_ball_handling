# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Script Purpose: Interactive verification of Robotera visuo-tactile sensing, dynamic grasping physics, and live OpenCV/Omniverse HUD telemetry.

Usage:
    python scripts/test_tactile_grasp.py --enable_cameras
"""

import argparse
import math
import time

# 1. Isaac Lab App Launcher
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Robotera fingertip tactile force sensing on dual balls.")
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
from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg

from robotera_ball_handling.scene_config import TactileGraspTestSceneCfg
from robotera_ball_handling.sensors import (
    apply_realsense_d405_noise,
    compute_visuo_tactile_contact_features,
)


def render_ascii_taxel_map(force_matrix_8x8: torch.Tensor, threshold: float = 0.02) -> str:
    """Renders an 8x8 force matrix with active contact patterns and numeric values (Left: Thumb, Right: Middle, Top: Tip)."""
    mirrored = force_matrix_8x8
    lines = []
    max_val = mirrored.max().item()
    has_contact = (max_val > 0.01)
    if not has_contact:
        for _ in range(8):
            lines.append("       # # # # # # # #   [  -    -    -    -    -    -    -    -  ]")
        return "\n".join(lines) + "\n       (Pad in open air / no contact)"

    # Isolate true localized contact area (above 25% of peak contact pressure)
    contact_cutoff = max(threshold, max_val * 0.25)
    for r in range(8):
        row_str = "       "
        for c in range(8):
            val = mirrored[r, c].item()
            if val > contact_cutoff:
                row_str += "* "
            else:
                row_str += "# "
        # Numerical forces in Newtons for each taxel in this row
        num_str = "   [" + " ".join([f"{mirrored[r, c].item():4.2f}N" if mirrored[r, c].item() > 0.01 else "   -  " for c in range(8)]) + "]"
        lines.append(row_str + num_str)
    return "\n".join(lines)


try:
    import omni.ui as ui
    OMNI_UI_AVAILABLE = True
except ImportError:
    OMNI_UI_AVAILABLE = False


SELECTED_FINGER_IDX = None
GLOBAL_TACTILE_HUD = None
OPENCV_MOUSE_SETUP = False
WINDOW_NAME = "Robotera Multi-Modal Cameras (Tactile Sensors & D405 Wrist)"


def select_finger(idx: int):
    """Sets the featured finger or sensor to inspect (0..4 for fingers, 5 for D405 Wrist)."""
    global SELECTED_FINGER_IDX
    SELECTED_FINGER_IDX = idx


def on_opencv_mouse(event, x, y, flags, param):
    """Click on the top row of 6 cards (150px each) to select that finger or camera."""
    if event == cv2.EVENT_LBUTTONDOWN and y < 206:
        card_idx = int(x // 150)
        if 0 <= card_idx <= 5:
            select_finger(card_idx)


class OmniverseTactileHUD:
    """Native Omniverse UI Window with clickable finger / camera selector and live view."""
    def __init__(self, finger_names: list[str]):
        self.image_provider = None
        self.window = None
        self.finger_names = finger_names
        if OMNI_UI_AVAILABLE:
            try:
                self.window = ui.Window("Robotera Multi-Modal Cameras Live View", width=920, height=600)
                self.image_provider = ui.ByteImageProvider()
                with self.window.frame:
                    with ui.VStack(spacing=4):
                        with ui.HStack(height=26, spacing=6):
                            ui.Label("Select View:", width=100)
                            for idx, name in enumerate(finger_names):
                                ui.Button(name, clicked_fn=lambda i=idx: select_finger(i), width=85)
                            ui.Button("D405 Wrist", clicked_fn=lambda: select_finger(5), width=110)
                        self.img_widget = ui.ImageWithProvider(self.image_provider, width=900, height=536)
                        def on_img_click(x, y, btn, mod):
                            if y < 206:
                                card_idx = int(x // 150)
                                if 0 <= card_idx <= 5:
                                    select_finger(card_idx)
                        try:
                            self.img_widget.set_mouse_pressed_fn(on_img_click)
                        except Exception:
                            pass
            except Exception:
                self.window = None
                self.image_provider = None

    def update(self, bgr_image: np.ndarray):
        if self.image_provider is not None:
            try:
                rgba = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGBA)
                h, w, _ = rgba.shape
                self.image_provider.set_bytes_data(rgba.flatten().tolist(), [w, h])
            except Exception:
                pass


# RealSense D405 Filter Flags are controlled directly in cameras.py (RealSenseD405NoiseModel.DEFAULT_ENABLE_*)
from robotera_ball_handling.sensors.cameras import RealSenseD405NoiseModel


def update_tactile_opencv_dashboard(
    tactile_depth: torch.Tensor,
    f_array: torch.Tensor,
    finger_names: list[str],
    index_sensor_idx: int,
    t_forces: torch.Tensor,
    p_forces: torch.Tensor,
    alpha: float,
    wrist_rgb: torch.Tensor | None = None,
    wrist_depth: torch.Tensor | None = None,
):
    """Renders a real-time HUD with dynamic finger and D405 wrist camera selection."""
    global GLOBAL_TACTILE_HUD, SELECTED_FINGER_IDX, OPENCV_MOUSE_SETUP
    if GLOBAL_TACTILE_HUD is None and OMNI_UI_AVAILABLE:
        GLOBAL_TACTILE_HUD = OmniverseTactileHUD(finger_names)

    # Default to Thumb if not yet selected
    if SELECTED_FINGER_IDX is None or SELECTED_FINGER_IDX > 5:
        SELECTED_FINGER_IDX = finger_names.index("Thumb") if "Thumb" in finger_names else index_sensor_idx

    featured_idx = SELECTED_FINGER_IDX

    # Squeeze depth to (5, H, W)
    d_tensor = tactile_depth
    if d_tensor.ndim == 4 and d_tensor.shape[-1] == 1:
        d_tensor = d_tensor.squeeze(-1)
    if d_tensor.ndim == 4:
        d_tensor = d_tensor[0]
    depth_np = d_tensor.detach().cpu().numpy()

    # Process D405 wrist camera feed if provided
    d405_lock_pct = 0.0
    d405_bgr = None
    d405_col = None
    if wrist_rgb is not None and wrist_depth is not None:
        noisy_d405 = apply_realsense_d405_noise(wrist_rgb, wrist_depth)

        # Prepare D405 RGB
        w_rgb = wrist_rgb[0] if wrist_rgb.ndim == 4 else wrist_rgb
        w_rgb_np = w_rgb.detach().cpu().numpy()
        if w_rgb_np.max() <= 1.0 and w_rgb_np.dtype != np.uint8:
            w_rgb_u8 = (w_rgb_np * 255.0).astype(np.uint8)
        else:
            w_rgb_u8 = w_rgb_np.astype(np.uint8)
        if w_rgb_u8.shape[-1] == 4:
            d405_bgr = cv2.cvtColor(w_rgb_u8, cv2.COLOR_RGBA2BGR)
        else:
            d405_bgr = cv2.cvtColor(w_rgb_u8, cv2.COLOR_RGB2BGR)

        # Prepare D405 Depth
        w_d = noisy_d405[0] if noisy_d405.ndim == 4 else noisy_d405
        if w_d.ndim == 3 and w_d.shape[-1] == 1:
            w_d = w_d.squeeze(-1)
        w_d_np = w_d.detach().cpu().numpy().astype(np.float32)

        valid_mask = (w_d_np >= 0.07) & (w_d_np <= 0.50)
        d405_lock_pct = float(valid_mask.mean()) * 100.0

        d_clamped = np.clip(w_d_np, 0.07, 0.30)
        norm_d = (d_clamped - 0.07) / (0.30 - 0.07)
        u8_d = (norm_d * 255.0).astype(np.uint8)
        d405_col = cv2.applyColorMap(u8_d, cv2.COLORMAP_TURBO)
        d405_col[~valid_mask] = (0, 0, 0)

    # Typography constants
    UI_FONT = cv2.FONT_HERSHEY_SIMPLEX
    UI_SCALE = 0.44
    UI_THICK = 1
    UI_GRAY = (180, 180, 180)

    # 1. Top Strip: 6 mini-cards (150x170 each => 6 x 150 = 900 total width)
    card_w, card_h = 150, 170
    max_range = 0.015  # 15mm continuous visual tracking range
    cards = []
    for i, name in enumerate(finger_names):
        d_raw = depth_np[i]
        d_clean = np.where(np.isnan(d_raw), 0.0, d_raw)
        d_clean = np.where(np.isinf(d_clean), max_range, d_clean)
        d_clean = np.clip(d_clean, 0.0, max_range)
        d_i = d_clean

        p_val = p_forces[i].item()
        f_val = t_forces[i].item()

        prox = (max_range - d_i) / max_range
        u8 = (prox * 255.0).astype(np.uint8)
        col = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
        col_resized = cv2.resize(col, (card_w, card_h), interpolation=cv2.INTER_NEAREST)

        card = np.zeros((card_h + 36, card_w, 3), dtype=np.uint8)
        card[36:, :] = col_resized

        is_selected = (i == featured_idx)
        is_touching = (f_val > 0.01 or p_val > 0.01)

        if is_selected:
            border_color = (0, 255, 255)  # Cyan border for selected
            border_thick = 2
        elif is_touching:
            border_color = (0, 200, 0)
            border_thick = 2
        else:
            border_color = (60, 60, 60)
            border_thick = 1

        cv2.rectangle(card, (0, 0), (card_w - 1, card_h + 35), border_color, border_thick)

        min_d_mm = d_clean.min() * 1000.0
        label = f"{name}: {f_val:.1f}N" if is_touching else f"{name}: {min_d_mm:.0f}mm"
        cv2.putText(card, label, (6, 24), UI_FONT, UI_SCALE, UI_GRAY, UI_THICK)
        cards.append(card)

    # 6th Card: D405 Wrist Camera
    card_d405 = np.zeros((card_h + 36, card_w, 3), dtype=np.uint8)
    if d405_col is not None:
        d405_thumb = cv2.resize(d405_col, (card_w, card_h), interpolation=cv2.INTER_NEAREST)
        card_d405[36:, :] = d405_thumb
    is_d405_selected = (featured_idx == 5)
    border_d405 = (0, 255, 255) if is_d405_selected else (60, 60, 60)
    thick_d405 = 2 if is_d405_selected else 1
    cv2.rectangle(card_d405, (0, 0), (card_w - 1, card_h + 35), border_d405, thick_d405)
    cv2.putText(card_d405, f"D405: {d405_lock_pct:.0f}%", (6, 24), UI_FONT, UI_SCALE, (0, 255, 255) if is_d405_selected else UI_GRAY, UI_THICK)
    cards.append(card_d405)

    top_strip = np.hstack(cards)
    total_w = top_strip.shape[1]  # 6 * 150 = 900
    bot_h = 330

    # 2. Bottom Area: Either D405 Wrist View (RGB + Depth) OR Tactile Finger View
    if featured_idx == 5 and d405_bgr is not None and d405_col is not None:
        # D405 Dual View (450x330 RGB + 450x330 Depth = 900x330)
        panel_w = 450
        rgb_panel = cv2.resize(d405_bgr, (panel_w, bot_h), interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(rgb_panel, (0, 0), (panel_w - 1, bot_h - 1), (0, 255, 255), 2)
        cv2.putText(rgb_panel, "Intel RealSense D405: Left Imager RGB (1280x720)", (12, 26), UI_FONT, 0.44, (220, 220, 220), 1)

        depth_panel = cv2.resize(d405_col, (panel_w, bot_h), interpolation=cv2.INTER_NEAREST)
        flicker_str = "ON" if RealSenseD405NoiseModel.DEFAULT_ENABLE_EDGE_FLICKER else "OFF"
        voronoi_str = "ON" if RealSenseD405NoiseModel.DEFAULT_ENABLE_VORONOI_DROPOUT else "OFF"
        contrast_str = "ON" if RealSenseD405NoiseModel.DEFAULT_ENABLE_LOW_CONTRAST else "OFF"
        occl_str = "ON" if RealSenseD405NoiseModel.DEFAULT_ENABLE_STEREO_OCCLUSION else "OFF"
        cv2.putText(depth_panel, f"D405: Noise:{flicker_str} | Voronoi:{voronoi_str} | Contrast:{contrast_str} | Occl:{occl_str}", (12, 26), UI_FONT, 0.35, (0, 255, 255), 1)
        cv2.putText(depth_panel, "[Turbo] 7cm (Blue) -> 30cm (Red) | Black: RealSense Dropout", (12, bot_h - 14), UI_FONT, 0.38, (200, 200, 200), 1)

        bottom_strip = np.hstack([rgb_panel, depth_panel])
    else:
        # Visuo-Tactile Finger View (330 + 330 + 240 = 900)
        bot_size = 330
        safe_idx = min(featured_idx, len(finger_names) - 1)
        feat_raw = depth_np[safe_idx]
        feat_clean = np.where(np.isnan(feat_raw), 0.0, feat_raw)
        feat_clean = np.where(np.isinf(feat_clean), max_range, feat_clean)
        feat_clean = np.clip(feat_clean, 0.0, max_range)
        feat_d = feat_clean

        feat_prox = (max_range - feat_d) / max_range
        feat_col = cv2.applyColorMap((feat_prox * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        feat_large = cv2.resize(feat_col, (bot_size, bot_size), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(feat_large, (0, 0), (bot_size - 1, bot_size - 1), (0, 255, 255), 2)

        taxel_np = f_array[safe_idx].detach().cpu().numpy()
        peak_force = float(taxel_np.max()) if taxel_np.size > 0 else 0.0

        if peak_force > 0.01:
            taxel_norm = np.clip(taxel_np / peak_force, 0.0, 1.0)
        else:
            feat_d_8x8 = cv2.resize(feat_d, (8, 8), interpolation=cv2.INTER_AREA)
            taxel_norm = (max_range - feat_d_8x8) / max_range

        taxel_col = cv2.applyColorMap((taxel_norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        taxel_large = cv2.resize(taxel_col, (bot_size, bot_size), interpolation=cv2.INTER_NEAREST)

        cell_size = float(bot_size) / 8.0
        for g in range(1, 8):
            pos = int(g * cell_size)
            cv2.line(taxel_large, (pos, 0), (pos, bot_size), (40, 40, 40), 1)
            cv2.line(taxel_large, (0, pos), (bot_size, pos), (40, 40, 40), 1)

        if peak_force > 0.01:
            max_r, max_c = np.unravel_index(np.argmax(taxel_np), taxel_np.shape)
            x0 = int(max_c * cell_size)
            y0 = int(max_r * cell_size)
            x1 = int((max_c + 1) * cell_size)
            y1 = int((max_r + 1) * cell_size)
            cv2.rectangle(taxel_large, (x0 + 1, y0 + 1), (x1 - 1, y1 - 1), (255, 255, 255), 2)

        cv2.rectangle(taxel_large, (0, 0), (bot_size - 1, bot_size - 1), (255, 150, 0), 2)

        info_w = total_w - (bot_size * 2)  # 900 - 660 = 240
        info_panel = np.zeros((bot_size, info_w, 3), dtype=np.uint8)
        idx_f = t_forces[safe_idx].item()
        idx_p = p_forces[safe_idx].item()
        min_dist_mm = feat_clean.min() * 1000.0

        featured_name = finger_names[safe_idx]
        cv2.putText(info_panel, f"Finger: {featured_name}", (14, 32), UI_FONT, UI_SCALE, UI_GRAY, UI_THICK)
        cv2.putText(info_panel, f"Min Distance: {min_dist_mm:.1f} mm", (14, 70), UI_FONT, UI_SCALE, UI_GRAY, UI_THICK)
        cv2.putText(info_panel, f"Total Force:  {idx_f:.2f} N", (14, 110), UI_FONT, UI_SCALE, UI_GRAY, UI_THICK)

        is_touch = (idx_f > 0.01 or idx_p > 0.01)
        status_text = "Touch: ACTIVE" if is_touch else "Touch: INACTIVE"
        cv2.putText(info_panel, status_text, (14, 150), UI_FONT, UI_SCALE, UI_GRAY, UI_THICK)

        cv2.putText(info_panel, "Colormap:", (14, 215), UI_FONT, UI_SCALE, UI_GRAY, UI_THICK)
        cv2.putText(info_panel, "[RED]  0-3mm (Contact)", (14, 242), UI_FONT, UI_SCALE, (50, 50, 255), UI_THICK)
        cv2.putText(info_panel, "[YEL]  7mm Proximity", (14, 270), UI_FONT, UI_SCALE, (0, 255, 255), UI_THICK)
        cv2.putText(info_panel, "[BLUE] >15mm (Air)", (14, 298), UI_FONT, UI_SCALE, (255, 120, 0), UI_THICK)
        cv2.rectangle(info_panel, (0, 0), (info_w - 1, bot_size - 1), (80, 80, 80), 1)

        bottom_strip = np.hstack([feat_large, taxel_large, info_panel])

    dashboard = np.vstack([top_strip, bottom_strip])

    # 1. Update native Omniverse UI window
    if GLOBAL_TACTILE_HUD is not None:
        GLOBAL_TACTILE_HUD.update(dashboard)

    # 2. Try OpenCV window with click callback and keybindings
    try:
        cv2.imshow(WINDOW_NAME, dashboard)
        if not OPENCV_MOUSE_SETUP:
            cv2.setMouseCallback(WINDOW_NAME, on_opencv_mouse, finger_names)
            OPENCV_MOUSE_SETUP = True
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            select_finger(key - ord('1'))
        elif key in [ord('6'), ord('d'), ord('D')]:
            select_finger(5)
        elif key == ord('t') and "Thumb" in finger_names:
            select_finger(finger_names.index("Thumb"))
        elif key == ord('i') and "Index" in finger_names:
            select_finger(finger_names.index("Index"))
        elif key == ord('m') and "Middle" in finger_names:
            select_finger(finger_names.index("Middle"))
        elif key == ord('r') and "Ring" in finger_names:
            select_finger(finger_names.index("Ring"))
        elif key == ord('p') and "Pinky" in finger_names:
            select_finger(finger_names.index("Pinky"))
    except Exception:
        pass


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

    print("\n" + "=" * 78)
    print(" [INFO] Tactile Grasp & Contact Verification Test Running (Static Fixed Ball)!")
    print(" - Single Red Ball is STATIC & IMMOVABLE (kinematic, zero gravity) at (X=0.18, Y=-0.02, Z=0.418).")
    print(" - Ball will NOT fall, move, or roll away.")
    print(" - Execution & Simulation Loops:")
    print(f"     * High-Level Policy Loop : {policy_freq:5.1f} Hz  (Period: {policy_period_ms:5.1f} ms | Decimation: {decimation})")
    print(f"     * Low-Level Control Loop : {ctrl_freq:5.1f} Hz  (Period: {ctrl_period_ms:5.1f} ms | Joint PD & PhysX)")
    print(" - Camera Specifications:")
    print(f"     * Head Camera    (Overview D435) : Resolution {head_res}  | Target {target_fps:.0f} FPS")
    print(f"     * Wrist Camera   (Macro D405)    : Resolution {wrist_res}  | Target {target_fps:.0f} FPS")
    print(f"     * Tactile Sensor (5 Fingertips)  : Resolution {tactile_res}    | Target {target_fps:.0f} FPS")
    print(" - Sensor Body Order mapped from PhysX:")
    for i, (b_name, d_name) in enumerate(zip(sensor_body_names, finger_names)):
        print(f"     [Sensor Index {i}] -> {d_name:6s} ('{b_name}')")
    print(f" - Displaying 8x8 Tactile Map on Index Finger Pad (Sensor Index {index_sensor_idx})")
    print(" - Live OpenCV Visuo-Tactile HUD window launched alongside simulation")
    print(" - Press 'Space' in GUI to Pause/Play | Press Ctrl+C to exit")
    print("=" * 78 + "\n")

    while simulation_app.is_running():
        # Drive full range grasp trajectory: oscillating from 0.0 (completely flat) to 1.0 (deep grasp)
        alpha = 0.50 * (1.0 - math.cos(step * 0.025))
        cur_targets = target_pos.clone()
        for j_idx in range(len(robot.data.joint_names)):
            j_name = robot.data.joint_names[j_idx]
            if "thumb_bend" in j_name:
                cur_targets[:, j_idx] = 0.0 + alpha * 0.70  # Thumb adduction: 0.0 (flat) -> 0.70 rad (grasp)
            elif "thumb_rota" in j_name:
                cur_targets[:, j_idx] = 0.0 + alpha * 1.30  # Thumb flexion: 0.0 (flat) -> 1.30 rad (grasp)
            elif "joint1" in j_name:
                cur_targets[:, j_idx] = 0.0 + alpha * 1.80  # Proximal flexion: 0.0 (completely flat) -> 1.80 rad (curled)
            elif "joint2" in j_name:
                cur_targets[:, j_idx] = 0.0 + alpha * 1.80  # Distal flexion: 0.0 (completely flat) -> 1.80 rad (curled)
            elif "bend_joint" in j_name:
                cur_targets[:, j_idx] = 0.0                  # Neutral adduction/abduction

        robot.set_joint_position_target(cur_targets)

        # Step physics & rendering
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        step += 1

        # Process tactile sensors every frame
        tactile_depth = scene["tactile_cameras"].data.output["distance_to_image_plane"]
        physics_contact = torch.norm(scene["contact_forces"].data.net_forces_w, dim=-1)

        # Convert depth & contact forces into (5, 8, 8) force array and total forces
        force_array, total_forces = compute_visuo_tactile_contact_features(
            tactile_depth, contact_forces=physics_contact
        )

        # Flatten to 1D finger forces for single env: (5,)
        t_forces = total_forces.view(-1)
        f_array = force_array.view(5, 8, 8)
        p_forces = physics_contact.view(-1)

        # Render unified multi-modal OpenCV dashboard (Tactile fingers + D405 Wrist view)
        wrist_rgb = scene["wrist_camera"].data.output["rgb"]
        wrist_depth = scene["wrist_camera"].data.output["distance_to_image_plane"]
        update_tactile_opencv_dashboard(
            tactile_depth,
            f_array,
            finger_names,
            index_sensor_idx,
            t_forces,
            p_forces,
            alpha,
            wrist_rgb=wrist_rgb,
            wrist_depth=wrist_depth,
        )


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

    scene_cfg = TactileGraspTestSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()

    run_grasp_test(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        simulation_app.close()

