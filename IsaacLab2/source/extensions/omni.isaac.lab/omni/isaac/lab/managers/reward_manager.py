# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward manager for computing reward signals for a given world."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import RewardTermCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

from constraint_generation import ConstraintGenerator
import os
import json
import copy
from utils_geoconst import load_functions_from_txt
import open3d as o3d
import numpy as np
import subprocess
import transform_utils as T
import utils_geoconst as utils

class RewardManager(ManagerBase):
    """Manager for computing reward signals for a given world.

    The reward manager computes the total reward as a sum of the weighted reward terms. The reward
    terms are parsed from a nested config class containing the reward manger's settings and reward
    terms configuration.

    The reward terms are parsed from a config class containing the manager's settings and each term's
    parameters. Each reward term should instantiate the :class:`RewardTermCfg` class.

    .. note::

        The reward manager multiplies the reward term's ``weight``  with the time-step interval ``dt``
        of the environment. This is done to ensure that the computed reward terms are balanced with
        respect to the chosen time-step interval in the environment.

    """

    _env: ManagerBasedRLEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        """Initialize the reward manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, RewardTermCfg]``).
            env: The environment instance.
        """
        # create buffers to parse and store terms
        self._term_names: list[str] = list()
        self._term_cfgs: list[RewardTermCfg] = list()
        self._class_term_cfgs: list[RewardTermCfg] = list()

        # call the base class constructor (this will parse the terms config)
        super().__init__(cfg, env)
        # prepare extra info to store individual reward term information
        self._episode_sums = dict()
        for term_name in self._term_names:
            self._episode_sums[term_name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # create buffer for managing reward per environment
        self._reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Buffer which stores the current step reward for each term for each environment
        self._step_reward = torch.zeros((self.num_envs, len(self._term_names)), dtype=torch.float, device=self.device)

    def __str__(self) -> str:
        """Returns: A string representation for reward manager."""
        msg = f"<RewardManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Reward Terms"
        table.field_names = ["Index", "Name", "Weight"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Weight"] = "r"
        # add info on each term
        for index, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            table.add_row([index, name, term_cfg.weight])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active reward terms."""
        return self._term_names

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Returns the episodic sum of individual reward terms.

        Args:
            env_ids: The environment ids for which the episodic sum of
                individual reward terms is to be returned. Defaults to all the environment ids.

        Returns:
            Dictionary of episodic sum of individual reward terms.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # store information
        extras = {}
        for key in self._episode_sums.keys():
            # store information
            # r_1 + r_2 + ... + r_n
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self._env.max_episode_length_s
            # reset episodic sum
            self._episode_sums[key][env_ids] = 0.0
        # reset all the reward terms
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        # return logged information
        return extras

    def compute(self, dt: float) -> torch.Tensor:
        """Computes the reward signal as a weighted sum of individual terms.

        This function calls each reward term managed by the class and adds them to compute the net
        reward signal. It also updates the episodic sums corresponding to individual reward terms.

        Args:
            dt: The time-step interval of the environment.

        Returns:
            The net reward signal of shape (num_envs,).
        """
        # reset computation
        self._reward_buf[:] = 0.0
        # iterate over all the reward terms
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            # update total reward
            self._reward_buf += value
            # update episodic sum
            self._episode_sums[name] += value

            # Update current reward for this step.
            self._step_reward[:, self._term_names.index(name)] = value / dt

        return self._reward_buf

    """
    Operations - Term settings.
    """

    def set_term_cfg(self, term_name: str, cfg: RewardTermCfg):
        """Sets the configuration of the specified term into the manager.

        Args:
            term_name: The name of the reward term.
            cfg: The configuration for the reward term.

        Raises:
            ValueError: If the term name is not found.
        """
        if term_name not in self._term_names:
            raise ValueError(f"Reward term '{term_name}' not found.")
        # set the configuration
        self._term_cfgs[self._term_names.index(term_name)] = cfg

    def get_term_cfg(self, term_name: str) -> RewardTermCfg:
        """Gets the configuration for the specified term.

        Args:
            term_name: The name of the reward term.

        Returns:
            The configuration of the reward term.

        Raises:
            ValueError: If the term name is not found.
        """
        if term_name not in self._term_names:
            raise ValueError(f"Reward term '{term_name}' not found.")
        # return the configuration
        return self._term_cfgs[self._term_names.index(term_name)]

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """
        terms = []
        for idx, name in enumerate(self._term_names):
            terms.append((name, [self._step_reward[env_idx, idx].cpu().item()]))
        return terms

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the terms
        for term_name, term_cfg in cfg_items:
            # check for non config
            if term_cfg is None:
                continue
            # check for valid config type
            if not isinstance(term_cfg, RewardTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type RewardTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # check for valid weight type
            if not isinstance(term_cfg.weight, (float, int)):
                raise TypeError(
                    f"Weight for the term '{term_name}' is not of type float or int."
                    f" Received: '{type(term_cfg.weight)}'."
                )
            # resolve common parameters
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
            # add function to list
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            # check if the term is a class
            if isinstance(term_cfg.func, ManagerTermBase):
                self._class_term_cfgs.append(term_cfg)

class GeoConstRewardManager(RewardManager):
    def __init__(self, task_dir: str, task_description: str, cfg: object, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.task_description = task_description
        self.constraint_generator = ConstraintGenerator({"model": "chatgpt-4o-latest"})
        img = env.scene.sensors['tiled_camera'].data.output['rgb'][0].detach().cpu().numpy()
        self.constraint_generator.generate(img, self.task_description, rekep_program_dir=task_dir)
        # load metadata
        with open(os.path.join(task_dir, 'metadata.json'), 'r') as f:
            self.program_info = json.load(f)
        env.register_geometries(self.program_info['object_to_segment'], task_dir)
        self.env = env
        utils.ENV = env
        self.constraint_fns = {}
        functions_dict = {
            "get_point_cloud": self.get_point_cloud_wrapper(),
            "grasp": self.grasp_all_candidates_wrapper(),
            "release": self.release_wrapper(),
            "env": self,
            "np": np,
            "subprocess": subprocess,
            "o3d": o3d,
        }
        for stage in range(1, self.program_info['num_stage'] + 1):  # stage starts with 1
            stage_dict = dict()
            load_path = os.path.join(task_dir, f'stage_{stage}_subgoal_constraints.txt')
            if not os.path.exists(load_path):
                func, _ = [], []
            else:
                ret = load_functions_from_txt(load_path, functions_dict, return_code=True) 
                func, code = ret['func'], ret["code"]
                if "grasp" in code:
                    func[0] = self.grasp_cost_wrapper(func[0], func[1:])
            self.constraint_fns[stage] = func
        self.batch_idx = 0
        ee_pos_b = self.env.get_ee_pos_b()
        bs = ee_pos_b.shape[0]
        self.stage_idx = np.array([1 for _ in range(bs)])

    def get_point_cloud_wrapper(self,):
        def get_point_cloud(part_name, ts):
            return self.env.part_to_pts_dict[ts][part_name][self.batch_idx]
        return get_point_cloud

    def release_wrapper(self,):
        def release():
            self.release()
        return release

    def release(self,):
        pass

    def calculate_reward(self, cost, stage_idx):
        reward = 1 / (cost + 1) + stage_idx * 2
        return reward

    def get_next_stage_idx(self, cost, current_stage_idx):
        if cost < 0.05:
            return current_stage_idx + 1
        return current_stage_idx
    
    def grasp_cost_wrapper(self, grasp_func, constraint_fns):
        name = grasp_func()
        # grasp_poses = candidates['subgoal_poses']
        # grasp_dict = select_grasp_with_constraints(grasp_poses, constraint_fns)
        # grasp_pose, approach, binormal = grasp_dict["grasp_pose"], grasp_dict['approach'], grasp_dict['binormal']
        get_point_cloud = self.get_point_cloud_wrapper()
        def grasp_cost():
            ee_pos = get_point_cloud("the gripper of the robot", -1)
            grasp_center = get_point_cloud(name, -1).mean(0)
            pos_cost = np.linalg.norm(ee_pos - grasp_center)
            cost = pos_cost 
            for constraint_fn in constraint_fns:
                cost += constraint_fn()
            # print("ee_pos:", ee_pos, "grasp_center:", grasp_center, "cost:", cost)
            return cost
        return grasp_cost

    def grasp_all_candidates_wrapper(self,):
        def grasp_all_candidates(name):
            ## TODO: only use the center of the grasping part to supervise the ee pos
            return name
            batch_idx = self.batch_idx
            env = self.env
            if name == "":
                return {
                    "subgoal_poses": env.get_ee_pose()[None, ...],
                }
            get_point_cloud = self.get_point_cloud_wrapper()
            segm_pts_3d = get_point_cloud(name, -1)
            rgbs = env.last_cam_obs[1]['rgb']
            pts_3d = env.last_cam_obs[1]['points']
            pcd_debug = o3d.geometry.PointCloud()
            pcd_debug.points = o3d.utility.Vector3dVector(pts_3d.reshape(-1, 3))
            pcd_debug.colors = o3d.utility.Vector3dVector(rgbs.reshape(-1, 3) / 255.)
            pcs_mean = segm_pts_3d.mean(0)
            segm_pts_3d -= pcs_mean
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(segm_pts_3d)
            pcd.colors = o3d.utility.Vector3dVector(np.ones((segm_pts_3d.shape[0], 3)))
            o3d.io.write_point_cloud("tmp.pcd", pcd)
            pcd.points = o3d.utility.Vector3dVector(segm_pts_3d + pcs_mean)
            o3d.io.write_point_cloud("tmp.ply", pcd)
            grasp_cfg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../gpd/cfg/eigen_params.cfg")
            grasp_bin_path = "detect_grasps"
            output = subprocess.check_output(['{}'.format(grasp_bin_path), '{}'.format(grasp_cfg_path), "tmp.pcd"])
            app_strs = str(output).split("Approach")[1:]
            approaches = []
            for app_str in app_strs:
                app_str = app_str.strip().split(':')[1].strip()
                app_vec =  app_str.split("\\n")
                app_vec = np.array([float(app_vec[0]), float(app_vec[1]), float(app_vec[2])])
                approaches.append(app_vec)
            approaches = np.stack(approaches, axis=0)
            pos_str = app_strs[-1]
            pos_strs = pos_str.split("Position")[1:]
            positions = []
            for pos_str in pos_strs:
                pos_str = pos_str.strip().split(':')[1].strip()
                pos_vec =  pos_str.split("\\n")
                pos_vec = np.array([float(pos_vec[0]), float(pos_vec[1]), float(pos_vec[2])])
                positions.append(pos_vec)
            positions = np.stack(positions, axis=0)

            binormal_str = pos_strs[-1]
            binormal_strs = binormal_str.split("Binormal")[1:]
            binormals = []
            for binormal_str in binormal_strs:
                binormal_str = binormal_str.strip().split(':')[1].strip()
                binormal_vec =  binormal_str.split("\\n")
                binormal_vec = np.array([float(binormal_vec[0]), float(binormal_vec[1]), float(binormal_vec[2])])
                binormals.append(binormal_vec)
            binormals = np.stack(binormals, axis=0)

            approach0 = env.APPROACH0.astype(np.float32)
            approach0 /= np.linalg.norm(approach0)
            binormal0 = env.BINORMAL0.astype(np.float32)
            binormal0 /= np.linalg.norm(binormal0)

            approaches = -approaches
            starts = positions + pcs_mean + 0.03 * approaches
            target_quats = []
            transform_mats = []

            pcd_debug = o3d.geometry.PointCloud()

            for i in range(len(approaches)):
                approach = approaches[i]
                binormal = binormals[i]

                source_points = np.stack([approach0, binormal0, np.array([0,0,0])], axis=0)
                target_points = np.stack([approach, binormal, np.array([0,0,0])], axis=0)
                transform_mat0 = np.identity(3)
                transform_mat =  cv2.estimateAffine3D(source_points, target_points, force_rotation=True)[0][:3, :3]
                transform_mats.append(transform_mat)
                mat = transform_mat @ transform_mat0
                target_quat = R.from_matrix(mat).as_quat()
                target_quats.append(target_quat)

            target_quats = np.stack(target_quats, axis=0)
            target_positions = starts
            subgoal_poses = np.concatenate([target_positions, target_quats], axis=-1)
            return {
                "subgoal_poses": subgoal_poses,
                "pcd_debug": pcd_debug
            }
        return grasp_all_candidates


    def compute(self, dt: float) -> torch.Tensor:
        """Computes the reward signal as a weighted sum of individual terms.

        This function calls each reward term managed by the class and adds them to compute the net
        reward signal. It also updates the episodic sums corresponding to individual reward terms.

        Args:
            dt: The time-step interval of the environment.

        Returns:
            The net reward signal of shape (num_envs,).
        """
        # reset computation
        part_to_pts_dict_last = copy.deepcopy(self.env.part_to_pts_dict)
        self.env.update_part_to_pts_dict()
        self._reward_buf[:] = 0.0
        ee_pos_b = self.env.get_ee_pos_b()
        bs = ee_pos_b.shape[0]
        for idx in range(bs):
            cost = 0
            self.batch_idx = idx
            stage_idx = self.stage_idx[idx]
            constraint_fns = self.constraint_fns[stage_idx]
            for constraint_fn in constraint_fns:
                a_cost = constraint_fn()
                cost += a_cost
            cost /= max(len(constraint_fns), 1)
            reward = self.calculate_reward(cost, stage_idx)
            # self._reward_buf[idx] = reward * dt
            self._reward_buf[idx] = -cost * 10
            next_stage_idx = self.get_next_stage_idx(cost, stage_idx)
            self.stage_idx[idx] = next_stage_idx
        self.env.part_to_pts_dict = copy.deepcopy(part_to_pts_dict_last)
        reward_buf = self._reward_buf
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            if name == "action_rate" or name == "joint_vel":
                value = term_cfg.func(self.env, **term_cfg.params) * term_cfg.weight * dt
                self._reward_buf += value
                self._step_reward[:, self._term_names.index(name)] = value / dt
            else:
                self._step_reward[:, self._term_names.index(name)] = reward_buf / dt
        return self._reward_buf
