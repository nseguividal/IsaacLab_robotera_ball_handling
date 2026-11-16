# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration template for the ball rotation task.

What goes here:
1. `RoboteraBallRotationSceneCfg`: Ground, lights, robot, task objects (e.g. balls),
   and sensors (cameras, contact sensors, tactile sensors).
2. `ActionsCfg`: Action terms mapping policy outputs to robot joint/actuator targets.
3. `ObservationsCfg`: Policy (and optional critic) observation groups/terms.
4. `RewardsCfg`: Reward terms and weights shaping the desired behavior.
5. `TerminationsCfg`: Episode timeout and failure/drop termination conditions.
6. `EventCfg`: Reset logic and domain randomization (mass, friction, actuator gains, etc.).
7. `RoboteraBallRotationEnvCfg`: Top-level `ManagerBasedRLEnvCfg` tying all sections
   together with simulation timestep (`dt`), `decimation`, and physics settings.
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# TODO: Import robot/sensor configs and custom MDP terms when implementing the task.
# from robotera_ball_handling.robots import ...
# from robotera_ball_handling.sensors import ...
# from . import mdp


# ==============================================================================
# 1. Scene Configuration
# ==============================================================================
@configclass
class RoboteraBallRotationSceneCfg(InteractiveSceneCfg):
    """Scene entities: ground, lighting, robot, objects, and sensors."""

    # TODO: Add ground, dome_light, robot, manipulation objects, and sensors.
    pass


# ==============================================================================
# 2. Action Configuration
# ==============================================================================
@configclass
class ActionsCfg:
    """Action space definitions (e.g., joint position targets)."""

    # TODO: Add ActionTermCfg fields.
    pass


# ==============================================================================
# 3. Observation Configuration
# ==============================================================================
@configclass
class ObservationsCfg:
    """Observation groups and terms fed to the RL policy."""

    # TODO: Define PolicyCfg(ObsGroup) with ObsTerm fields.
    pass


# ==============================================================================
# 4. Rewards Configuration
# ==============================================================================
@configclass
class RewardsCfg:
    """Reward terms and weights."""

    # TODO: Add RewTerm fields using isaaclab.envs.mdp or local `mdp` functions.
    pass


# ==============================================================================
# 5. Terminations Configuration
# ==============================================================================
@configclass
class TerminationsCfg:
    """Episode termination conditions (timeouts, drops, out-of-bounds)."""

    # TODO: Add DoneTerm fields.
    pass


# ==============================================================================
# 6. Events & Domain Randomization Configuration
# ==============================================================================
@configclass
class EventCfg:
    """Startup/reset events and domain randomization terms."""

    # TODO: Add EventTerm fields.
    pass


# ==============================================================================
# 7. Top-Level Environment Configuration
# ==============================================================================
@configclass
class RoboteraBallRotationEnvCfg(ManagerBasedRLEnvCfg):
    """Complete ManagerBasedRLEnvCfg for the ball rotation task."""

    scene: RoboteraBallRotationSceneCfg = RoboteraBallRotationSceneCfg(num_envs=64, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Configure simulation dt, decimation, episode length, and physics."""
        # TODO: Set self.decimation, self.episode_length_s, and self.sim (SimulationCfg).
        pass
