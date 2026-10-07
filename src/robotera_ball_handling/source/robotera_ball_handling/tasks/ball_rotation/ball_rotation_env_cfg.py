# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Robotera Hand Dual-Ball Rotation environment."""

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# Sensor & Robot configs
from robotera_ball_handling.robots import ROBOTERA_XHAND_RIGHT_CFG
from robotera_ball_handling.sensors import (
    FINGERTIP_CONTACT_SENSOR_CFG,
    REALSENSE_D405_WRIST_CFG,
    REALSENSE_D435_OVERVIEW_CFG,
    VISUO_TACTILE_FINGERTIPS_CFG,
)

# Task MDP components
from . import mdp


# ==============================================================================
# 1. Scene Configuration with Newton Contact Physics & Multi-Sensory Array
# ==============================================================================
@configclass
class RoboteraBallRotationSceneCfg(InteractiveSceneCfg):
    """Scene definition with hand, 2 balls, cameras, and tactile sensors."""

    # Ground Plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2500.0,
            color=(0.95, 0.95, 0.95),
        ),
    )

    # Robotera XHand (Right)
    robot: ArticulationCfg = ROBOTERA_XHAND_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Ball 1 (Red manipulation sphere - radius 24mm, mass 50g)
    ball1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball1",
        spawn=sim_utils.SphereCfg(
            radius=0.024,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2), roughness=0.3),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.85,
                dynamic_friction=0.70,
                restitution=0.05,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.080, -0.022, 0.440),
        ),
    )

    # Ball 2 (Blue manipulation sphere - radius 24mm, mass 50g)
    ball2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball2",
        spawn=sim_utils.SphereCfg(
            radius=0.024,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.9), roughness=0.3),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.85,
                dynamic_friction=0.70,
                restitution=0.05,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.080, 0.022, 0.440),
        ),
    )

    # Multi-Modal Sensors
    wrist_camera: TiledCameraCfg = REALSENSE_D405_WRIST_CFG
    overview_camera: TiledCameraCfg = REALSENSE_D435_OVERVIEW_CFG
    tactile_cameras: TiledCameraCfg = VISUO_TACTILE_FINGERTIPS_CFG
    contact_forces: ContactSensorCfg = FINGERTIP_CONTACT_SENSOR_CFG


# ==============================================================================
# 2. Action Configuration (12-DoF Joint Position Setpoints)
# ==============================================================================
@configclass
class ActionsCfg:
    """Action spaces for finger joint position targets."""

    finger_positions = ActionTerm(
        class_type=isaac_mdp.JointPositionAction,
        asset_name="robot",
        joint_names=["right_hand_.*_joint.*"],
        scale=0.25,
        use_default_offset=True,
    )


# ==============================================================================
# 3. Observation Configuration (Proprioception + Tactile + Goal)
# ==============================================================================
@configclass
class ObservationsCfg:
    """Observation spaces for RL policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations fed to the actor-critic network."""

        # Hand proprioception
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})

        # Dual ball 6D states relative to palm
        ball_state = ObsTerm(func=mdp.dual_ball_state)

        # Fingertip contact force magnitudes (physics engine)
        contact_forces = ObsTerm(
            func=isaac_mdp.contact_sensor_net_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces")},
        )

        # Visuo-tactile 5-fingertip force array (optical gel deformation to force matrix)
        visuo_tactile_forces = ObsTerm(
            func=mdp.visuo_tactile_force_array_obs,
            params={"sensor_name": "tactile_cameras", "taxels": (8, 8)},
        )

        # Actions history
        last_action = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ==============================================================================
# 4. Rewards Configuration
# ==============================================================================
@configclass
class RewardsCfg:
    """Reward terms encouraging continuous dual-ball rotation."""

    # Primary Task: Rotate the two balls around each other (target: 1.5 rad/s)
    rotation_velocity = RewTerm(
        func=mdp.dual_ball_rotation_velocity,
        weight=2.5,
        params={"target_velocity": 1.5},
    )

    # Secondary: Keep balls centered in the palm cradle
    containment = RewTerm(
        func=mdp.dual_ball_containment,
        weight=1.0,
        params={"std": 0.05},
    )

    # Keep the two balls in proximity
    proximity = RewTerm(
        func=mdp.dual_ball_proximity,
        weight=0.5,
        params={"std": 0.02},
    )

    # Penalize jerky high-frequency joint actions
    action_rate = RewTerm(
        func=isaac_mdp.action_rate_l2,
        weight=-0.01,
    )

    # Penalize excessive joint velocities
    joint_vel = RewTerm(
        func=isaac_mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# ==============================================================================
# 5. Terminations Configuration
# ==============================================================================
@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)

    ball_drop = DoneTerm(
        func=mdp.ball_dropped,
        params={"min_height": 0.32, "max_dist_from_palm": 0.12},
    )


# ==============================================================================
# 6. Complete Environment Configuration with Newton Physics
# ==============================================================================
@configclass
class RoboteraBallRotationEnvCfg(ManagerBasedRLEnvCfg):
    """Complete environment configuration for Robotera Ball Handling."""

    # Scene & Entities
    scene: RoboteraBallRotationSceneCfg = RoboteraBallRotationSceneCfg(num_envs=64, env_spacing=2.5)

    # MDP Components
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Configure simulation timestep and Newton / GPU Contact Physics engine."""
        # General simulation parameters
        self.decimation = 2  # Policy runs at 30 Hz (physics at 60 Hz)
        self.episode_length_s = 15.0

        # Simulation Physics Setup
        self.sim = SimulationCfg(
            dt=1.0 / 60.0,
            render_interval=self.decimation,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.85,
                dynamic_friction=0.70,
                restitution=0.05,
            ),
        )
