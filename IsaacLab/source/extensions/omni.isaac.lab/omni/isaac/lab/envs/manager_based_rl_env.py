# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import torch
from collections.abc import Sequence
from typing import Any, ClassVar

from omni.isaac.version import get_version

from omni.isaac.lab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
from omni.isaac.lab.ui.widgets import ManagerLiveVisualizer

from .common import VecEnvStepReturn
from .manager_based_env import ManagerBasedEnv, ManagerCameraBasedEnv
from .manager_based_rl_env_cfg import ManagerBasedRLEnvCfg


class ManagerBasedRLEnv(ManagerBasedEnv, gym.Env):
    """The superclass for the manager-based workflow reinforcement learning-based environments.

    This class inherits from :class:`ManagerBasedEnv` and implements the core functionality for
    reinforcement learning-based environments. It is designed to be used with any RL
    library. The class is designed to be used with vectorized environments, i.e., the
    environment is expected to be run in parallel with multiple sub-environments. The
    number of sub-environments is specified using the ``num_envs``.

    Each observation from the environment is a batch of observations for each sub-
    environments. The method :meth:`step` is also expected to receive a batch of actions
    for each sub-environment.

    While the environment itself is implemented as a vectorized environment, we do not
    inherit from :class:`gym.vector.VectorEnv`. This is mainly because the class adds
    various methods (for wait and asynchronous updates) which are not required.
    Additionally, each RL library typically has its own definition for a vectorized
    environment. Thus, to reduce complexity, we directly use the :class:`gym.Env` over
    here and leave it up to library-defined wrappers to take care of wrapping this
    environment for their agents.

    Note:
        For vectorized environments, it is recommended to **only** call the :meth:`reset`
        method once before the first call to :meth:`step`, i.e. after the environment is created.
        After that, the :meth:`step` function handles the reset of terminated sub-environments.
        This is because the simulator does not support resetting individual sub-environments
        in a vectorized environment.

    """

    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: ManagerBasedRLEnvCfg
    """Configuration for the environment."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
        """
        # initialize the base class to setup the scene.
        super().__init__(cfg=cfg)
        # store the render mode
        self.render_mode = render_mode

        # initialize data and constants
        # -- counter for curriculum
        self.common_step_counter = 0
        # -- init buffers
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # -- set the framerate of the gym video recorder wrapper so that the playback speed of the produced video matches the simulation
        self.metadata["render_fps"] = 1 / self.step_dt

        print("[INFO]: Completed setting up the environment...")

    """
    Properties.
    """

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps."""
        return math.ceil(self.max_episode_length_s / self.step_dt)

    """
    Operations - Setup.
    """

    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # -- command manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # call the parent class to load the managers for observations and actions.
        super().load_managers()

        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def setup_manager_visualizers(self):
        """Creates live visualizers for manager terms."""

        self.manager_visualizers = {
            "action_manager": ManagerLiveVisualizer(manager=self.action_manager),
            "observation_manager": ManagerLiveVisualizer(manager=self.observation_manager),
            "command_manager": ManagerLiveVisualizer(manager=self.command_manager),
            "termination_manager": ManagerLiveVisualizer(manager=self.termination_manager),
            "reward_manager": ManagerLiveVisualizer(manager=self.reward_manager),
            "curriculum_manager": ManagerLiveVisualizer(manager=self.curriculum_manager),
        }

    """
    Operations - MDP
    """

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """Run rendering without stepping through the physics.

        By convention, if mode is:

        - **human**: Render to the current display and return nothing. Usually for human consumption.
        - **rgb_array**: Return an numpy.ndarray with shape (x, y, 3), representing RGB values for an
          x-by-y pixel image, suitable for turning into a video.

        Args:
            recompute: Whether to force a render even if the simulator has already rendered the scene.
                Defaults to False.

        Returns:
            The rendered image as a numpy array if mode is "rgb_array". Otherwise, returns None.

        Raises:
            RuntimeError: If mode is set to "rgb_data" and simulation render mode does not support it.
                In this case, the simulation render mode must be set to ``RenderMode.PARTIAL_RENDERING``
                or ``RenderMode.FULL_RENDERING``.
            NotImplementedError: If an unsupported rendering mode is specified.
        """
        # run a rendering step of the simulator
        # if we have rtx sensors, we do not need to render again sin
        if not self.sim.has_rtx_sensors() and not recompute:
            self.sim.render()
        # decide the rendering mode
        if self.render_mode == "human" or self.render_mode is None:
            return None
        elif self.render_mode == "rgb_array":
            # check that if any render could have happened
            if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
                raise RuntimeError(
                    f"Cannot render '{self.render_mode}' when the simulation render mode is"
                    f" '{self.sim.render_mode.name}'. Please set the simulation render mode to:"
                    f"'{self.sim.RenderMode.PARTIAL_RENDERING.name}' or '{self.sim.RenderMode.FULL_RENDERING.name}'."
                    " If running headless, make sure --enable_cameras is set."
                )
            # create the annotator if it does not exist
            if not hasattr(self, "_rgb_annotator"):
                import omni.replicator.core as rep

                # create render product
                self._render_product = rep.create.render_product(
                    self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
                )
                # create rgb annotator -- used to read data from the render product
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            # note: initially the renerer is warming up and returns empty data
            if rgb_data.size == 0:
                return np.zeros((self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3), dtype=np.uint8)
            else:
                return rgb_data[:, :, :3]
        else:
            raise NotImplementedError(
                f"Render mode '{self.render_mode}' is not supported. Please use: {self.metadata['render_modes']}."
            )

    def close(self):
        if not self._is_closed:
            # destructor is order-sensitive
            del self.command_manager
            del self.reward_manager
            del self.termination_manager
            del self.curriculum_manager
            # call the parent class to close the environment
            super().close()

    """
    Helper functions.
    """

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.single_observation_space = gym.spaces.Dict()
        for group_name, group_term_names in self.observation_manager.active_terms.items():
            # extract quantities about the group
            has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
            group_dim = self.observation_manager.group_obs_dim[group_name]
            # check if group is concatenated or not
            # if not concatenated, then we need to add each term separately as a dictionary
            if has_concatenated_obs:
                self.single_observation_space[group_name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=group_dim)
            else:
                self.single_observation_space[group_name] = gym.spaces.Dict({
                    term_name: gym.spaces.Box(low=-np.inf, high=np.inf, shape=term_dim)
                    for term_name, term_dim in zip(group_term_names, group_dim)
                })
        # action space (unbounded since we don't impose any limits)
        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # update the curriculum for environments that need a reset
        self.curriculum_manager.compute(env_ids=env_ids)
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)
        # apply events such as randomizations for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- rewards manager
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- curriculum manager
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- command manager
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- termination manager
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- recorder manager
        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0

class ManagerCameraBasedRLEnv(ManagerCameraBasedEnv, ManagerBasedRLEnv):
    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: ManagerBasedRLEnvCfg
    """Configuration for the environment."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
        """
        # initialize the base class to setup the scene.
        ManagerCameraBasedEnv.__init__(self, cfg=cfg)

        # store the render mode
        self.render_mode = render_mode

        # initialize data and constants
        # -- counter for curriculum
        self.common_step_counter = 0
        # -- init buffers
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # -- set the framerate of the gym video recorder wrapper so that the playback speed of the produced video matches the simulation
        self.metadata["render_fps"] = 1 / self.step_dt

        print("[INFO]: Completed setting up the environment...")

    
    def get_cam_obs(self, ):
        camera = self.scene.sensors['tiled_camera']
        rgb_b = camera.data.output['rgb'].detach().cpu().numpy()
        depth_b = camera.data.output['depth']
        instrinsic_matrice_b = camera.data.intrinsic_matrices
        point_cloud_b = unproject_depth(camera.data.output["distance_to_image_plane"], instrinsic_matrice_b)
        point_cloud_b = transform_points(point_cloud_b, camera.data.pos_w, camera.data.quat_w_ros)
        bs, h, w = rgb_b.shape[0], rgb_b.shape[1], rgb_b.shape[2]
        point_cloud_b = point_cloud_b.reshape(bs, h, w, 3)
        point_cloud_b = point_cloud_b.detach().cpu().numpy()
        obs = {
            "rgb": rgb_b,
            "points": point_cloud_b
        }
        self.last_cam_obs = obs
        return obs



class GeoManipRLEnv(ManagerCameraBasedRLEnv):

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
        """
        # initialize the base class to setup the scene.
        super().__init__(cfg=cfg)

    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # -- command manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # call the parent class to load the managers for observations and actions.
        super().load_managers()

        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        self.reward_manager = GeoConstRewardManager(self.cfg.task_dir, self.cfg.task_description, self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def get_ee_pos_b(self, ):
        ee_pos_b = (self.robot_b._data.body_pos_w[:, -2, :] + self.robot_b._data.body_pos_w[:, -1, :] / 2.)
        ee_pos_b = ee_pos_b.detach().cpu().numpy()
        return ee_pos_b
    
    def get_cam_obs(self, ):
        camera = self.scene.sensors['tiled_camera']
        rgb_b = camera.data.output['rgb'].detach().cpu().numpy()
        depth_b = camera.data.output['depth']
        instrinsic_matrice_b = camera.data.intrinsic_matrices
        point_cloud_b = unproject_depth(camera.data.output["distance_to_image_plane"], instrinsic_matrice_b)
        point_cloud_b = transform_points(point_cloud_b, camera.data.pos_w, camera.data.quat_w_ros)
        bs, h, w = rgb_b.shape[0], rgb_b.shape[1], rgb_b.shape[2]
        point_cloud_b = point_cloud_b.reshape(bs, h, w, 3)
        point_cloud_b = point_cloud_b.detach().cpu().numpy()
        obs = {
            "rgb": rgb_b,
            "points": point_cloud_b
        }
        self.last_cam_obs = obs
        return obs

    def segment_0(self, obj_part, task_dir=None):
        rgb_b = self.get_cam_obs()['rgb']
        rgb_0 = rgb_b[0]
        cv2.imwrite("rgb_obs.png", rgb_0[:, :, ::-1])
        mask_0 = self.segmentor.segment(image_path="rgb_obs.png", obj_description=obj_part, rekep_program_dir=task_dir)
        return mask_0
    
    def mask_to_pcs_0(self, mask_0):
        points_b = self.get_cam_obs()['points']
        points_0 = points_b[0]
        points_0 = points_0[mask_0 > 0]
        return points_0

    def register_geometries(self, part_lists, task_dir):
        self.part_lists = part_lists
        self.robot_b = self.scene['robot']
        self.robot_body_pos_b = self.robot_b._data.body_pos_w
        self.part_to_pts_dict = []
        self.segmentor = Segmentor({
            "box_threshold": 0.1,
            "text_threshold":  0.05,
            "margin_ratio": 0.1,
            "temperature": 0.3,
            "top_p": 0.2,
        })

        part_name_to_prim_0 = {}
        part_name_to_prim_vert_index_0 = {}
        self.part_lists = part_lists

        if not hasattr(self, "all_prim_points_0"):
            all_prim_points_0 = []
            all_prims = []
            all_prim_points_indices_0 = []
            all_prim_indices_0 = []
            prims = [x for x in self.scene.stage.Traverse() if "/World/envs/env_0" in str(x.GetPrimPath()) and "visuals" in str(x.GetPrimPath()).lower() and "panda" not in str(x.GetPrimPath()).lower() and str(x.GetTypeName())=="Xform"]
            idx = 0
            for prim in prims:
                if "table" in str(prim.GetPrimPath()).lower():
                    mesh_path = str(prim.GetPrimPath()) + "/TableGeom"
                else:
                    mesh_path = str(prim.GetPrimPath()) + "/visuals"
                p = self.scene.stage.GetPrimAtPath(mesh_path)
                if not p.IsValid():
                    print("invalid prim path")
                    import ipdb;ipdb.set_trace()
                transform = np.asarray(omni.usd.get_world_transform_matrix(p))
                prim_points_0 = np.array(p.GetAttribute("points").Get())
                prim_points_0 = np.concatenate([prim_points_0, np.ones((len(prim_points_0), 1))], axis=-1)
                prim_points_0 = prim_points_0 @ transform
                prim_points_0 = prim_points_0[..., :3]
                all_prims += [p for _ in range(len(prim_points_0))]
                all_prim_points_0.append(prim_points_0)
                all_prim_points_indices_0.append(np.arange(len(prim_points_0)))
                all_prim_indices_0 += [idx for _ in range(len(prim_points_0))]
                idx += 1
            all_prim_points_0 = np.concatenate(all_prim_points_0, axis=0)
            all_prim_points_indices_0 = np.concatenate(all_prim_points_indices_0, axis=0)
            self.all_prim_points_indices_0 = all_prim_points_indices_0
            self.all_prim_points_0 = all_prim_points_0
            self.all_prims = np.array(all_prims)
            self.all_prim_indices_0 = np.array(all_prim_indices_0)
        for idx, obj_part in enumerate(part_lists):
            if "gripper" in obj_part:
                pass
            else:
                mask_0 = self.segment_0(obj_part, task_dir)
                pts_0 = self.mask_to_pcs_0(mask_0)
                dists_0 = np.linalg.norm(self.all_prim_points_0 - pts_0.mean(0), axis=-1)
                index_0 = np.argmin(dists_0)
                prim_0 = self.all_prims[index_0]
                prim_0_index = self.all_prim_indices_0[index_0]
                prim_points_0 = self.all_prim_points_0[self.all_prim_indices_0 == prim_0_index]
                dists = np.linalg.norm(prim_points_0[None, ...] - pts_0[:, None, ...], axis=-1)
                prim_vert_index = np.argmin(dists, axis=-1)
            part_name_to_prim_0[obj_part] = prim_0
            part_name_to_prim_vert_index_0[obj_part] = prim_vert_index
        self.part_name_to_prim_0 = part_name_to_prim_0
        self.part_name_to_prim_vert_index_0 = part_name_to_prim_vert_index_0
        self.update_part_to_pts_dict()
    
    def update_part_to_pts_dict(self,):
        part_lists = self.part_lists
        part_to_pts_dict_latest = {}
        pts_b = self.get_ee_pos_b()
        bs = pts_b.shape[0]
        for part_name in part_lists:
            if "gripper" in part_name:
                if "approach" in part_name:
                    start_b = self.get_ee_pos_b()
                    approach_b = self.robot_b._data.body_pos_w[:, -3, :] - (self.robot_b._data.body_pos_w[:, -2, :] + self.robot_b._data.body_pos_w[:, -1, :]) / 2.
                    approach_b = approach_b / torch.norm(approach_b, dim=-1)[:, None]
                    approach_b = approach_b.detach().cpu().numpy()
                    end_b = start_b + approach_b * 0.3
                    pts_b = np.linspace(start_b, end_b, 5, axis=1)
                elif "binormal" in part_name:
                    start_b = self.get_ee_pos_b()
                    binormal_b = self.robot_b._data.body_pos_w[:, -1, :] - self.robot_b._data.body_pos_w[:, -2, :]
                    binormal_b = binormal_b / torch.norm(binormal_b, dim=-1)[:, None]
                    binormal_b = binormal_b.detach().cpu().numpy()
                    end_b = start_b + binormal_b * 0.05
                    pts_b = np.linspace(start_b, end_b, 5, axis=1)
                else:
                    pts_b = self.get_ee_pos_b()[:, None, ...]
                part_to_pts_dict_latest[part_name] = pts_b
            else:
                prim_0 = self.part_name_to_prim_0[part_name]
                prim_vert_index_0 = self.part_name_to_prim_vert_index_0[part_name]
                prim_path_0 = str(prim_0.GetPath())
                prim_points_b = []
                for idx in range(bs):
                    prim_path = prim_path_0.replace("env_0", "env_{}".format(idx))
                    prim = self.scene.stage.GetPrimAtPath(prim_path)
                    prim_points = np.array(prim.GetAttribute("points").Get())[prim_vert_index_0]
                    transform = np.asarray(omni.usd.get_world_transform_matrix(prim))
                    prim_points = np.concatenate([prim_points, np.ones((len(prim_points), 1))], axis=-1)
                    prim_points = prim_points @ transform
                    prim_points = prim_points[..., :3]
                    prim_points_b.append(prim_points)
                prim_points_b = np.stack(prim_points_b, axis=0)
                part_to_pts_dict_latest[part_name] = prim_points_b
        self.part_to_pts_dict.append(part_to_pts_dict_latest)

    def get_part_to_pts_dict(self,):
        return self.part_to_pts_dict