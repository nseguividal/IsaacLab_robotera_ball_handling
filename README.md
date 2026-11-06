# UFactory uf850 + Robotera XHand Manipulation (Isaac Lab)

NVIDIA Isaac Lab simulation and reinforcement learning platform combining the **UFactory uf850 (6-DoF Robotic Arm)** with the **Robotera XHand (12-DoF Dexterous Hand)** into a unified 18-DoF manipulation system on a tabletop workspace.

## Git Branches

- **`ufactory-arm-and-hand`** *(this branch)*: UFactory uf850 arm bolted to the table with Robotera XHand palm mounted directly to the tool flange (18 active DoFs unified articulation).
- **`hand-ball-handling`**: Standalone hand platform focused on in-hand dual-ball (Baoding) continuous rotation.

---

## Features

- **Unified Articulation (18 DoFs)**:
  - **UFactory uf850 Arm**: 6 revolute joints (`joint1` to `joint6`) with high-stiffness control ($K_p=800$, $K_d=60$).
  - **Robotera XHand**: 12 compliant finger joints ($K_p=15$, $K_d=0.5$). Redundant wrist pitch/roll joints are removed so the palm base plate (`right_hand_base_link`) bolts flush directly to the arm flange (`link6`).
  - **Table Mounting**: Arm base bolted firmly at `pos=(-0.10, 0.0, 0.32)` on top of the oak table surface.
- **Sensors & Perception**:
  - **RealSense D405**: Wrist macro RGB-D camera with physical stereo noise modeling (stereo parallax occlusion, edge flicker, low-contrast dropouts, and spatial filtering).
  - **RealSense D435**: Static overview POV RGB-D camera.
  - **Visuo-Tactile Sensors**: 5x $(12 \times 10)$ taxel arrays (120 taxels per finger, 600 total) with 270° lateral pad encirclement and Maxwell viscoelastic creep relaxation.
  - **PhysX Contact Sensors**: 3D contact force vectors and normal forces on fingertips.
- **Sim-to-Real Features**:
  - Hardware bus latency queue (33.3 ms delay).
  - Domain randomization: object mass, surface friction, actuator gains, and pose perturbations.

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

### 1. Test Arm + Hand Coordination on Table
Interactive simulation verifying arm reach over the table and coordinated finger grasping with live telemetry:
```bash
python src/robotera_ball_handling/scripts/test_arm_hand.py --enable_cameras
```

### 2. Test Tactile Grasp & 12x10 Pressure HUD
Interactive simulation displaying the live OpenCV tactile pressure matrix HUD (12x10 grid per finger) and RealSense camera view:
```bash
python src/robotera_ball_handling/scripts/test_tactile_grasp.py --enable_cameras
```

### 3. Individual Finger Motion Diagnostic
Verifies joint motion and tactile sensor readouts on each finger sequentially:
```bash
python src/robotera_ball_handling/scripts/test_individual_fingers.py --enable_cameras
```

### 4. Train RL Policy (RSL-RL PPO)
Launch headless reinforcement learning training:
```bash
python src/robotera_ball_handling/scripts/rsl_rl/train.py --headless --num_envs 64
```

### 5. Play Trained Policy
Visualize and evaluate a trained policy checkpoint:
```bash
python src/robotera_ball_handling/scripts/rsl_rl/play.py --num_envs 16
```

---

## Repository Structure

```
isaaclab_projects/
├── README.md               # guide with all the execution commands
├── Information.md          # documentation on the sim-to-real physics parameters & sensor tuning
└── src/robotera_ball_handling/
    ├── scripts/            # executable scripts & RL stuff
    │   ├── test_arm_hand.py        # UFactory arm + hand on table test
    │   ├── test_tactile_grasp.py   # visuotactile sensors + wrist depth camera with noise (robotera)
    │   ├── test_individual_fingers.py  # hardcoded sequence of fingers moving
    │   └── rsl_rl/                 # PPO train and play scripts
    └── source/robotera_ball_handling/
        ├── assets/         # robot URDFs, USD references, and 3D STL meshes
        │   └── robots/xhand/urdf/  # robotera hand urdf
        ├── robots/         # articulation definitions (ufactory_xhand.py, robotera_xhand.py)
        ├── sensors/        # sensor definitions: visuotactile & cameras
        ├── tasks/          # RL environments (MDP rewards, observations)
        └── scene_config.py # scene setup: table, room, robots, lightning 
```

