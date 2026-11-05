File to save information about the simulation: 

Sim-To-Real bridging:

## 1. Noise Modeling for the depth camera (DONE) 
- Added noise to the depth camera so it looks similar to the real world

Noise is modeled in the class: RealSenseD405NoiseModel (defined in cameras.py)

Simulates the physical optical and ASIC phenomena of the Intel RealSense D405:
    1. Low-Contrast / Textureless Stereo Dropout (Zero-depth holes)
    2. Stereo Occlusion Shadows (stereo baseline blind spots on silhouettes)
    3. Saturation Dropout
    4. Edge Flickering (Stereo correlation window bleeding)
    5. Hard Optical Min-Z (7 cm) and Max-Z (50 cm) Search Limits
    6. RealSense Spatial Filter & Inter-Finger Depth Fading



## 2. Hand Joint Friction & Damping 
- Tuned finger joint friction and viscous damping to make finger movements more realistic:
  1. Joint Friction: `<dynamics friction="0.4" />` (in URDF, tuned from 1.0 to 0.4)
  2. Joint Viscous Damping: `<dynamics damping="0.5" />` (in URDF, tuned from 1.0 to 0.5)
  3. Wrist mount link given explicit inertial to prevent PhysX root joint warnings.

## 2.1 Scene & Room Isolation (DONE)
- Isolated room interior from external world elements in `scene_config.py`:
  1. Deactivated `/World/Room/GroundPlane`: Removes the external physics ground plane and collision mesh.
  2. Deactivated `/World/Room/DomeLight`: Removes the infinite exterior world HDRI/sky dome that showed behind the walls.
  3. Deactivated `/World/Room/table_low_327`: Removes redundant low table so only the designated oak table is present.
  4. Preserved: Room floor, wood/panel walls, ceiling lamps, and ceiling `RectLight` illumination.


## 3. Tactile and Fingertips (DONE)
- Modeled realistic elastomeric visuo-tactile behavior matching real Robotera XHand tactile hardware:
  1. **12x10 Taxel Grid (120 Taxels) with 270° Encirclement:**
     - Real XHand features a curved tactile skin wrapping 270° around each distal phalanx with 12 longitudinal rows and 10 circumferential columns.
     - Upgraded camera aperture to 3.8 and resolution to 20x24 (pooled 2:1 to 12 rows x 10 cols) to capture the wide 270° lateral encirclement.
  2. **Elastomeric Membrane Contact Gating & Pressure Profile (`tactile.py`):**
     - In open air (`in_contact == False`): Opaque pad is undeformed (0 mm indentation, 0 N force, dark purple/blue background).
     - Under contact (`in_contact == True`):
       * `posinf` (rays missing the object into empty air) represents uncompressed pad (`depth = gel_thickness`, 0 mm indentation).
       * Compressed rays (< 3mm) produce realistic pressure gradients (Red at center -> Yellow -> Blue).
       * Near-clipping penetration (`nan` / `< 0.1mm`) maps to deepest indentation (`0.0 mm` / Red).
  3. **Rubber Physics Material (`scene_config.py`):**
     - Applied compliant `FINGERTIP_MATERIAL_CFG` (high friction: static 2.0 / dynamic 1.5, low restitution 0.0, compliant stiffness 2000.0, damping 50.0).
  4. **Viscoelastic Relaxation Creep & Drift (`TactileSignalDynamics` in `tactile.py`):**
     - Maxwell/Kelvin-Voigt stress relaxation creep under sustained contact: 8% decay over 2.0s time constant.
     - Mean-reverting Ornstein-Uhlenbeck random walk baseline drift (+/- 0.025 N) and analog taxel jitter.

## 4. Control Latency (DONE)
- Simulates real robot communication bus (CAN / EtherCAT / USB) and motor driver filtering latency:
  1. Added a 2-step FIFO command buffer (`deque(maxlen=2)`) in `scripts/test_tactile_grasp.py`.
  2. Introduces ~33.3 ms delay at 60 Hz control loop, aligning simulated policy commands with physical actuator response.

## 5. Domain Randomization (DONE)
- Integrated into `RoboteraBallRotationEnvCfg` via `EventCfg` (`tasks/ball_rotation/ball_rotation_env_cfg.py`):
  1. **Object Mass:** Uniform randomization between 35g and 65g (nominal 50g +/- 30%).
  2. **Friction:** Randomization of static friction (0.50 - 1.20) and dynamic friction (0.40 - 1.00) between balls and fingertips across 32 buckets.
  3. **Motor Jitter / Actuator Gains:** Scale stiffness and damping by +/- 15% on episode reset.
  4. **Initial Pose Jitter:** Scaled joint perturbations (+/- 10%) on environment reset.





## Newton physics 
- Change to Newton physics, but for that we need to use the new isaaclab version
