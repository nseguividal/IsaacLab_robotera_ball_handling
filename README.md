# Robotera Dexterous In-Hand Manipulation (Isaac Lab)

NVIDIA Isaac Lab simulation and reinforcement learning framework for dexterous in-hand manipulation using the **Robotera XHand (12-DoF Dexterous Hand)**.

## Features

- **Robot Model**: Robotera XHand (12 active finger DoFs + 2 wrist DoFs) with URDF and collision meshes.
- **Sensors & Perception**:
  - **RealSense D405**: Wrist-mounted macro RGB-D camera with realistic noise modeling (stereo occlusion, edge flicker, dropout).
  - **RealSense D435**: Static overhead POV RGB-D camera.
  - **Visuo-Tactile Sensors**: 5x $(12 \times 10)$ taxel arrays (120 taxels per finger, 600 total) with 270° lateral encirclement.
  - **PhysX Contact Sensors**: 3D contact force vectors and normal forces on fingertips.
- **Sim-to-Real Bridging**:
  - Hardware communication latency queue (33.3 ms delay).
  - Elastomeric pad viscoelastic creep relaxation and baseline drift.
  - Domain randomization: ball mass, finger friction, actuator gains, and initial pose perturbations.
- **Tasks**:
  - `Isaac-Robotera-Ball-Rotation-v0`: In-hand dual Baoding ball continuous rotation task.

---

## Setup & Installation

1. Activate your Isaac Lab conda environment:
```bash
conda activate isaaclab
```

2. Install the `robotera_ball_handling` package in editable mode:
```bash
pip install -e src/robotera_ball_handling
```

---

## How to Run

Run commands from the repository root (`isaaclab_projects`):

### 1. Test Tactile Grasp & 12x10 Pressure HUD
Interactive simulation with live OpenCV tactile HUD (shows 12x10 taxel pressure grid and D405 wrist view):
```bash
python src/robotera_ball_handling/scripts/test_tactile_grasp.py --enable_cameras
```

### 2. Test RealSense Multi-Camera Preview
Visualizes overview and wrist cameras with live noise filters:
```bash
python src/robotera_ball_handling/scripts/visualize_hand_cameras.py --enable_cameras
```

### 3. Individual Finger Motion Diagnostic
Tests motion and tactile force readout on each finger one by one:
```bash
python src/robotera_ball_handling/scripts/test_individual_fingers.py --enable_cameras
```

### 4. Train with RSL-RL (PPO)
Launch headless reinforcement learning training:
```bash
python src/robotera_ball_handling/scripts/rsl_rl/train.py --headless --num_envs 64
```

### 5. Play Trained Policy
Run and visualize a trained policy checkpoint:
```bash
python src/robotera_ball_handling/scripts/rsl_rl/play.py --num_envs 16
```

---

## Repository Structure

```
isaaclab_projects/
├── README.md
├── Information.md          # Sim-to-real documentation & parameters
└── src/robotera_ball_handling/
    ├── scripts/            # Standalone test scripts & RL runners
    │   ├── test_tactile_grasp.py
    │   ├── test_individual_fingers.py
    │   ├── visualize_hand_cameras.py
    │   └── rsl_rl/         # PPO train and play scripts
    └── source/robotera_ball_handling/
        ├── assets/         # Robot URDFs and 3D STL meshes
        ├── robots/         # Hand articulation definitions
        ├── sensors/        # RealSense cameras & tactile models
        └── tasks/          # RL environments (MDP rewards, observations)
```

