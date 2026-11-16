# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MDP reward functions template.

What goes here:
- Functions with signature `(env: ManagerBasedRLEnv, ...) -> torch.Tensor`
  returning per-environment scalar reward tensors of shape `(env.num_envs,)`.
- Examples: task progress/velocity tracking, object containment, contact shaping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# TODO: Implement custom reward functions, e.g.:
# def my_reward(env: ManagerBasedRLEnv, ...) -> torch.Tensor:
#     return torch.zeros(env.num_envs, device=env.device)
