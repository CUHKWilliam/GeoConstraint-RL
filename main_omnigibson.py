import torch
import numpy as np
import json
import os
import argparse
from environment import ReKepOGEnv 
from constraint_generation import ConstraintGenerator2
import transform_utils as T
from omnigibson.robots.fetch import Fetch
from utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
    grasp_all_candidates,
)
import cv2
import ipdb
import subprocess
import open3d as o3d
from segment import Segmentor, get_point_cloud
from scipy.spatial.transform import Rotation as R


class Main:
    def __init__(self, scene_file, config_path="./configs/config.yaml", visualize=False, cam_id=1):
        global_config = get_config(config_path=config_path)
        self.config = global_config['main']
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.visualize = visualize
        self.segmentor = Segmentor(global_config['segmentation'])
        # set random seed
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        # initialize keypoint proposer and constraint generator
        self.constraint_generator = ConstraintGenerator2(global_config['constraint_generator'])
        # initialize environment
        self.env = ReKepOGEnv(global_config['env'], scene_file, verbose=False)
        self.env.segmentor = self.segmentor
        self.env.previous_pose = self.env.get_ee_pose()
        self.grasp_state = 0
        self.cam_id = cam_id
        self.env.cam_id = cam_id
        self.release = self.release_wrapper()
        self.mask_to_pc = self.mask_to_pc_wrapper()
        self.set_object_attribute()
        
        
    def set_object_attribute(self, ):
        objs = self.env.og_env.scene.objects
        for obj in objs:
            if "mass" in obj.get_init_info()['args']:
                link_dict = obj.links
                link_dict["base_link"].mass = obj.get_init_info()['args']['mass']
            
    def generate_cost_fns(self, instruction, rekep_program_dir=None, hint=""):
        self.rekep_program_dir = rekep_program_dir
        self.env.reset()
        cam_obs = self.env.get_cam_obs()
        rgb = cam_obs[self.cam_id]['rgb']
        cv2.imwrite('debug.png', rgb)
        # import ipdb;ipdb.set_trace()
        points = cam_obs[self.cam_id]['points']
        mask = cam_obs[self.cam_id]['seg']

        # ====================================
        # = keypoint proposal and constraint generation
        # ====================================
        rekep_program_dir = self.constraint_generator.generate(rgb, instruction, rekep_program_dir=rekep_program_dir, hint=hint)
        print(f'{bcolors.HEADER}Constraints generated{bcolors.ENDC}')
        self.env.register_keypoints(self.program_info['object_to_segment'] + ["gripper"], rekep_program_dir)
        self.constraint_fns = dict()
        self.constraint_fns_code = dict()
        functions_dict = {
            "get_point_cloud": get_point_cloud,
            "grasp": grasp_all_candidates,
            "release": self.release,
            "env": self.env,
            "np": np,
            "subprocess": subprocess,
            "o3d": o3d,
            "mask_to_pc": self.mask_to_pc,
        }
        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
            self.program_info = json.load(f)
        for stage in range(1, self.program_info['num_stage'] + 1):  # stage starts with 1
            stage_dict = dict()
            stage_dict_code = dict()
            for constraint_type in ['subgoal', 'path']:
                load_path = os.path.join(rekep_program_dir, f'stage_{stage}_{constraint_type}_constraints.txt')
                if not os.path.exists(load_path):
                    func, code = [], []
                else:
                    ret = load_functions_from_txt(load_path, functions_dict, return_code=True) 
                    func, code = ret['func'], ret["code"]
                ## merge the target constraints and the sub-goal constraint
                stage_dict[constraint_type] = func
                stage_dict_code[constraint_type] = code
                if constraint_type == "path":
                    for func in stage_dict[constraint_type]:
                        self.path_constraint_state[str(func)] = 0 # set inactivate
            self.constraint_fns[stage] = stage_dict
            self.constraint_fns_code[stage] = stage_dict_code

    def train_model(self, ):
        pass

    def release_wrapper(self,):
        def release():
            self.env.close_gripper()
        return release

    def mask_to_pc_wrapper(self,):
        def mask_to_pc(mask):
            env = self.env 
            env.get_cam_obs()
            pcs = env.last_cam_obs[env.cam_id]['points'][mask]
            return pcs
        return mask_to_pc


    def register_moving_part_names(self, grasp=True):
        moving_part_names = []
        if grasp:
            code = self.constraint_fns_code[self.stage]['subgoal']
            ## set moving part the part connected to the end-effector
            moving_part_name = code.split('grasp("')[1].split('")')[0]
            moving_part_obj_name = moving_part_name.split("of")[-1].strip()
            for key in self.env.part_to_pts_dict[-1].keys():
                if "axis" in key or "frame" in key:
                    continue
                if key.split("of")[-1].strip() == moving_part_obj_name:
                    moving_part_names.append(key)
        for key in self.env.part_to_pts_dict[-1].keys():
            if "gripper" in key:
                moving_part_names.append(key)
        self.env.moving_part_names = moving_part_names

    def _update_stage(self, stage):
        # update stage
        self.stage = stage
        # clear action queue
        self.action_queue = []
        # update keypoint movable mask
        # self._update_keypoint_movable_mask()
        self.first_iter = True

    def _execute_grasp_action(self):
        if self.env.is_grasping:
            return
        pregrasp_pose = self.env.get_ee_pose()
        grasp_pose = pregrasp_pose.copy()
        grasp_pose[:3] += T.quat2mat(pregrasp_pose[3:]) @ np.array([self.config['grasp_depth'], 0, 0])
        grasp_action = np.concatenate([grasp_pose, [self.env.get_gripper_close_action()]])
        self.env.execute_action(grasp_action, precise=True)
        self.env.is_grasping = True
        self.register_moving_part_names(grasp=True)
    
    def _execute_release_action(self):
        if not self.env.is_grasping:
            return
        self.env.open_gripper()
        self.register_moving_part_names(grasp=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pen', help='task to perform')
    parser.add_argument('--use_cached_query', action='store_true', help='instead of querying the VLM, use the cached query')
    parser.add_argument('--apply_disturbance', action='store_true', help='apply disturbance to test the robustness')
    parser.add_argument('--visualize', action='store_true', help='visualize each solution before executing (NOTE: this is blocking and needs to press "ESC" to continue)')
    args = parser.parse_args()
    args.use_cached_query = True

    task_list = {
        'pen': {
            'scene_file': './configs/og_scene_file_pen.json',
            'instruction': 'put the pen perpendicularly into the black cup',
            'rekep_program_dir': './vlm_query/pen-4',
            'hint': "",
            "config_path": "./configs/config.yaml",
            },
        'fridge': {
            'scene_file': './configs/og_scene_file_fridge.json',
            'instruction': 'open the fridge',
            'rekep_program_dir': './vlm_query/fridge',
            'hint': "",
            "config_path": "./configs/config_fridge.yaml",
        },
        'trash_can': {
            'scene_file': './configs/og_scene_file_trash_can.json',
            'instruction': 'open the trash can',
            'rekep_program_dir': './vlm_query/trash_can',
            'hint': "",
            "config_path": "./configs/config.yaml",
        },
         'carrot': {
            'scene_file': './configs/og_scene_file_carrot.json',
            'instruction': 'cut the carrot with the knife',
            'rekep_program_dir': './vlm_query/carrot-6',
            'hint': "",
            "config_path": "./configs/config.yaml",
        },
        'keyboard': {
            'scene_file': './configs/og_scene_file_keyboard.json',
            'instruction': 'play the first 7 notes of song "little star" on the keyboard',
            # 'instruction': 'press the button',
            'rekep_program_dir': './vlm_query/keyboard',
            'hint': "",
            "cam_id": 3,
            "config_path": "./configs/config.yaml",
        },
        'computer keyboard': {
            'scene_file': './configs/og_scene_file_computer-keyboard.json',
            'instruction': 'type "hi" on the computer keyboard',
            'rekep_program_dir': './vlm_query/computer-keyboard-2',
            'hint': "close the gripper first",
            "cam_id": 2,
            "config_path": "./configs/config.yaml",
        },
        'drawer': {
            'scene_file': './configs/og_scene_file_drawer.json',
            'instruction': 'open the drawer',
            # 'instruction': 'press the button',
            'rekep_program_dir': './vlm_query/drawer',
            'hint': "the handle shifts along the axis of ",
            "config_path": "./configs/config.yaml",
        },
    }
    task = task_list['carrot']
    if "cam_id" in task.keys():
        cam_id = task["cam_id"]
    else:
        cam_id = 1
    scene_file = task['scene_file']
    instruction = task['instruction']
    hint = task['hint']
    config_path = task['config_path']
    os.makedirs(task['rekep_program_dir'], exist_ok=True)
    main = Main(scene_file, config_path=config_path, visualize=args.visualize, cam_id=cam_id)
    main.generate_cost_fns(instruction,
                    rekep_program_dir=task['rekep_program_dir'] if args.use_cached_query else None,
                    hint=hint)
    main.train_model()
    
    