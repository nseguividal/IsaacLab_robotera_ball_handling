# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""RL agent / runner configuration template (RSL-RL PPO).

What goes here:
- Training runner settings: `num_steps_per_env`, `max_iterations`, `save_interval`,
  `experiment_name`, `empirical_normalization`.
- `policy`: Actor-critic network architecture (`RslRlPpoActorCriticCfg`).
- `algorithm`: PPO hyperparameters (`RslRlPpoAlgorithmCfg`: learning rate,
  clipping, entropy coef, GAE lambda/gamma, etc.).
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RoboteraBallRotationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO runner configuration template."""

    experiment_name = "robotera_ball_rotation"

    # TODO: Tune runner, policy network, and PPO algorithm hyperparameters.
    policy = RslRlPpoActorCriticCfg()
    algorithm = RslRlPpoAlgorithmCfg()
