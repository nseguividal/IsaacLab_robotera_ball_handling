# Robotera Dexterous In-Hand Manipulation (Isaac Lab)

This repository contains an NVIDIA Isaac Lab simulation and reinforcement learning framework for dexterous in-hand manipulation using the **Robotera XHand (12-DoF Dexterous Hand)**.

## Key Features

- **Robot Model**: Robotera XHand (12 active DoFs + 2 wrist DoFs) with URDF and STL mesh assets.
- **Multi-Modal Perception**:
  - **RealSense D405**: Wrist-mounted macro RGB-D camera.
  - **RealSense D435**: Static head/overhead POV RGB-D camera.
  - **Visuo-Tactile Optical Arrays**: 5x $(8 \times 8)$ spatial normal pressure matrices on fingertips modeled after Touchlab / GelSight tactile sensors.
  - **PhysX Contact Sensors**: Direct 3D normal force vector reporting on distal phalanges.
- **Tasks**:
  - `Isaac-Robotera-Ball-Rotation-v0`: In-hand dual Baoding ball continuous rotation task.
- **Physics**: Newton / GPU Contact physics solver in Isaac Lab.
- **RL Pipeline**: RSL-RL (PPO) training and evaluation workflows.

---

## Quick Start & Verification Scripts

### 1. Interactive Visualizer with RealSense Cameras
```bash
python src/robotera_ball_handling/scripts/visualize_hand_cameras.py --enable_cameras
```

### 2. Tactile Grasp & $(8 \times 8)$ Pressure Matrix Test
```bash
python src/robotera_ball_handling/scripts/test_tactile_grasp.py --enable_cameras
```

### 3. Individual Finger Motion & Sensor Alignment Diagnostic
```bash
python src/robotera_ball_handling/scripts/test_individual_fingers.py --enable_cameras
```

### 4. RSL-RL PPO Training
```bash
python src/robotera_ball_handling/scripts/rsl_rl/train.py --headless --num_envs 64
```

---

## Repository Structure

```
isaaclab_projects/
├── README.md
├── .gitignore
└── src/
    └── robotera_ball_handling/
        ├── config/
        │   └── extension.toml
        ├── scripts/
        │   ├── test_individual_fingers.py
        │   ├── test_tactile_grasp.py
        │   ├── visualize_hand_cameras.py
        │   └── rsl_rl/
        │       ├── train.py
        │       └── play.py
        ├── setup.py
        └── source/
            └── robotera_ball_handling/
                ├── assets/ (URDFs & STL meshes)
                ├── robots/ (Robotera XHand Articulation configs)
                ├── sensors/ (RealSense D405/D435 & Tactile arrays)
                └── tasks/
                    └── ball_rotation/ (MDP rewards, observations, PPO configs)
```

## License
BSD-3-Clause
