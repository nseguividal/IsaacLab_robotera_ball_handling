# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MDP observation functions template.

What goes here:
- Functions with signature `(env: ManagerBasedRLEnv, ...) -> torch.Tensor`
  returning per-environment observation tensors of shape `(env.num_envs, obs_dim)`.
- Examples: object pose/velocity relative to palm, tactile/contact feature extraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# TODO: Implement custom observation functions, e.g.:
# def my_observation(env: ManagerBasedRLEnv, ...) -> torch.Tensor:
#     return torch.zeros((env.num_envs, 1), device=env.device)
