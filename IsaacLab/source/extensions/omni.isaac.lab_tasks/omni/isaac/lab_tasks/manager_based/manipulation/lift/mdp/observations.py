# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms
import cv2
import numpy as np

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_link_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_link_state_w[:, :3], robot.data.root_link_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b

def get_rgb(
        env:ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    rgbs = env.scene.sensors['tiled_camera'].data.output['rgb'] / 255.
    ## TODO:
    cv2.imwrite('debug.png', (rgbs[0] * 255).detach().cpu().numpy().astype(np.uint8)[:, :, ::-1])
    mean_tensor = torch.mean(rgbs, dim=(1, 2), keepdim=True)
    rgbs -= mean_tensor
    return rgbs
