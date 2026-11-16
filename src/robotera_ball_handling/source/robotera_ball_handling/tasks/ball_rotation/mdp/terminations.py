# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MDP termination functions template.

What goes here:
- Functions with signature `(env: ManagerBasedRLEnv, ...) -> torch.Tensor`
  returning boolean tensors of shape `(env.num_envs,)` indicating which
  environments should terminate/reset.
- Examples: object dropped below minimum height, object out of workspace bounds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# TODO: Implement custom termination functions, e.g.:
# def my_termination(env: ManagerBasedRLEnv, ...) -> torch.Tensor:
#     return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
