import base64
from openai import OpenAI
import os
import cv2
import json
import parse
import numpy as np
import time
from datetime import datetime
import re
from utils import query_vlm_model

TEMPERATURE = 0.6
TOP_P=0.1
# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class ConstraintGenerator:
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.vlm_model = config['model'] if 'model' in config.keys() else "chatgpu-4o-latest"
        self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './vlm_query')
        with open(os.path.join(self.base_dir, 'prompt_template.txt'), 'r') as f:
            self.prompt_template = f.read()

    def _build_prompt(self, image_path, instruction):
        img_base64 = encode_image(image_path)
        prompt_text = self.prompt_template.format(instruction)
        # save prompt
        with open(os.path.join(self.task_dir, 'prompt.txt'), 'w') as f:
            f.write(prompt_text)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt_template.format(instruction=instruction)
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    },
                ]
            }
        ]
        return messages
    
    def _parse_other_metadata(self, output):
        data_dict = dict()
        # find num_stages
        num_stages_template = "num_stages = {num_stages}"
        for line in output.split("\n"):
            num_stages = parse.parse(num_stages_template, line)
            if num_stages is not None:
                break
        if num_stages is None:
            raise ValueError("num_stages not found in output")
        data_dict['num_stages'] = int(num_stages['num_stages'])
        # find grasp_keypoints
        grasp_keypoints_template = "grasp_keypoints = {grasp_keypoints}"
        for line in output.split("\n"):
            grasp_keypoints = parse.parse(grasp_keypoints_template, line)
            if grasp_keypoints is not None:
                break
        if grasp_keypoints is None:
            raise ValueError("grasp_keypoints not found in output")
        # convert into list of ints
        grasp_keypoints = grasp_keypoints['grasp_keypoints'].replace("[", "").replace("]", "").split(",")
        grasp_keypoints = [int(x.strip()) for x in grasp_keypoints]
        data_dict['grasp_keypoints'] = grasp_keypoints
        # find release_keypoints
        release_keypoints_template = "release_keypoints = {release_keypoints}"
        for line in output.split("\n"):
            release_keypoints = parse.parse(release_keypoints_template, line)
            if release_keypoints is not None:
                break
        if release_keypoints is None:
            raise ValueError("release_keypoints not found in output")
        # convert into list of ints
        release_keypoints = release_keypoints['release_keypoints'].replace("[", "").replace("]", "").split(",")
        release_keypoints = [int(x.strip()) for x in release_keypoints]
        data_dict['release_keypoints'] = release_keypoints
        return data_dict

    def _save_metadata(self, metadata):
        for k, v in metadata.items():
            if isinstance(v, np.ndarray):
                metadata[k] = v.tolist()
        with open(os.path.join(self.task_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        print(f"Metadata saved to {os.path.join(self.task_dir, 'metadata.json')}")

    def generate(self, img, instruction, metadata, task_dir=None):
        """
        Args:
            img (np.ndarray): image of the scene (H, W, 3) uint8
            instruction (str): instruction for the query
        Returns:
            save_dir (str): directory where the constraints
        """
        # create a directory for the task
        fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + instruction.lower().replace(" ", "_")
        if task_dir is None:
            self.task_dir = os.path.join(self.base_dir, fname)
            os.makedirs(self.task_dir, exist_ok=True)
        # save query image
        image_path = os.path.join(self.task_dir, 'query_img.png')
        cv2.imwrite(image_path, img[..., ::-1])
        # build prompt
        messages = self._build_prompt(image_path, instruction)
        # stream back the response
        stream = query_vlm_model(self.client, self.config['model'], messages, TEMPERATURE, TOP_P, stream=True)
        output = ""
        start = time.time()
        for chunk in stream:
            print(f'[{time.time()-start:.2f}s] Querying OpenAI API...', end='\r')
            if chunk.choices[0].delta.content is not None:
                output += chunk.choices[0].delta.content
        print(f'[{time.time()-start:.2f}s] Querying OpenAI API...Done')
        # save raw output
        with open(os.path.join(self.task_dir, 'output_raw.txt'), 'w') as f:
            f.write(output)
        # parse and save constraints
        self._parse_and_save_constraints(output, self.task_dir)
        # save metadata
        metadata.update(self._parse_other_metadata(output))
        self._save_metadata(metadata)
        return self.task_dir


def _parse_and_save_constraints(output, save_dir):
        # parse into function blocks
        lines = output.split("\n")
        functions = dict()
        meta_data = {}
        max_stage = -1
        objects_to_segment = []
        for i, line in enumerate(lines):
            if line.strip().startswith("def"):
                start = i
                name = line.split("(")[0].split("def ")[1]
                stage = int(name.split("_")[1])
                if stage > max_stage:
                    max_stage = stage
            if line.strip().startswith("return"):
                end = i
                functions[name] = lines[start:end+1]
            # if line.strip().startswith("object_to_segment"):
            #     ret = {}
            #     exec("".join(lines[i:]).replace("`", ""), {}, ret)
            #     objects_to_segment = ret['object_to_segment']
            if "get_point_cloud" in line:
                obj = line.split("\"")[1]
                objects_to_segment.append(obj)
            if "grasp(\"" in line:
                obj = line.split("\"")[1]
                if obj.strip() != "":
                    objects_to_segment.append(obj)
        objects_to_segment = list(set(objects_to_segment))
        objects_to_segment += [
            "the gripper of the robot",
            "the gripper approach of the robot",
            "the gripper binormal of the robot",
        ]
        meta_data.update({
            "num_stage": max_stage,
            "object_to_segment": objects_to_segment
        })
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(meta_data, f)
        # organize them based on hierarchy in function names
        groupings = dict()
        for name in functions:
            parts = name.split("_")[:-1]  # last one is the constraint idx
            key = "_".join(parts)
            if key not in groupings:
                groupings[key] = []
            groupings[key].append(name)
        # save them into files
        for key in groupings:
            with open(os.path.join(save_dir, f"{key}_constraints.txt"), "w") as f:
                for name in groupings[key]:
                    f.write("\n".join(functions[name]) + "\n\n")
        print(f"Constraints saved to {save_dir}")

class ConstraintGenerator2:
    def __init__(self, config, prompt_template_path=None):
        self.config = config
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './vlm_query')
        self.vlm_model = config['model'] if 'model' in config.keys() else "chatgpu-4o-latest"
        if prompt_template_path is None:
            prompt_template_path = os.path.join(self.base_dir, 'prompt_template.txt')
        with open(prompt_template_path, 'r') as f:
            self.prompt_template = f.read()
        self.query_history = []

    def _build_prompt_cost_functions(self, stage_num=None, prompt_text_only=False):
        prompt_cost_function = self.prompt_template.split("<STEP SPLITTER>")[1]
        if stage_num is not None:
            prompt_cost_function = prompt_cost_function.replace("for each stage", "for ONLY the {} stage".format(stage_num))
        with open("./vlm_query/geometry_knowledge.txt", "r") as f:
            geometry_knowledge = f.read()
        prompt_text = prompt_cost_function.format(geometry_knowledge)
        if prompt_text_only:
            return prompt_text
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text
                },
            ]
        }
        return message

    def _build_prompt_geometry_constraints(self, image_path, instruction, hint="", prompt_text_only=False):
        prompt_geometry_constraints = self.prompt_template.split("<STEP SPLITTER>")[0]
        with open("./vlm_query/geometry_constraints_prompt.txt", 'r') as f:
            goemetry_constranits_prompt = f.read()
        prompt_text = prompt_geometry_constraints.format(instruction, goemetry_constranits_prompt)
        if prompt_text_only:
            return prompt_text
        img_base64 = encode_image(image_path)
        message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    },
                ]
            }
        
        return message
        
    def _parse_other_metadata(self, output):
        data_dict = dict()
        # find num_stages
        num_stages_template = "num_stages = {num_stages}"
        for line in output.split("\n"):
            num_stages = parse.parse(num_stages_template, line)
            if num_stages is not None:
                break
        if num_stages is None:
            raise ValueError("num_stages not found in output")
        data_dict['num_stages'] = int(num_stages['num_stages'])
        # find grasp_keypoints
        grasp_keypoints_template = "grasp_keypoints = {grasp_keypoints}"
        for line in output.split("\n"):
            grasp_keypoints = parse.parse(grasp_keypoints_template, line)
            if grasp_keypoints is not None:
                break
        if grasp_keypoints is None:
            raise ValueError("grasp_keypoints not found in output")
        # convert into list of ints
        grasp_keypoints = grasp_keypoints['grasp_keypoints'].replace("[", "").replace("]", "").split(",")
        grasp_keypoints = [int(x.strip()) for x in grasp_keypoints]
        data_dict['grasp_keypoints'] = grasp_keypoints
        # find release_keypoints
        release_keypoints_template = "release_keypoints = {release_keypoints}"
        for line in output.split("\n"):
            release_keypoints = parse.parse(release_keypoints_template, line)
            if release_keypoints is not None:
                break
        if release_keypoints is None:
            raise ValueError("release_keypoints not found in output")
        # convert into list of ints
        release_keypoints = release_keypoints['release_keypoints'].replace("[", "").replace("]", "").split(",")
        release_keypoints = [int(x.strip()) for x in release_keypoints]
        data_dict['release_keypoints'] = release_keypoints
        return data_dict

    def _save_metadata(self, metadata):
        for k, v in metadata.items():
            if isinstance(v, np.ndarray):
                metadata[k] = v.tolist()
        with open(os.path.join(self.task_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        print(f"Metadata saved to {os.path.join(self.task_dir, 'metadata.json')}")

    def generate_geometry_constraints(self, image_path, instruction, rekep_program_dir=None, hint="", seed=None, override=False):
        prompt = self._build_prompt_geometry_constraints(image_path, instruction + ". DETAILS: {}.".format(hint), hint)
        self.query_history.append(prompt)
        output_constraint_file = os.path.join(rekep_program_dir, "output_constraints.txt")
        self.task_dir = rekep_program_dir
        if not os.path.exists(output_constraint_file) or override:
            # stream back the response
            stream = query_vlm_model(self.client, self.vlm_model, self.query_history, TEMPERATURE, TOP_P, stream=True)
            output = ""
            start = time.time()
            for chunk in stream:
                print(f'[{time.time()-start:.2f}s] Querying OpenAI API...', end='\r')
                if chunk.choices[0].delta.content is not None:
                    output += chunk.choices[0].delta.content
            print(f'[{time.time()-start:.2f}s] Querying OpenAI API...Done')
            with open(output_constraint_file, "w") as f:
                f.write(output)
        else:
            with open(output_constraint_file, "r") as f:
                output = f.read()
            self.query_history.append(
                {"role": "system", "content": "{}".format(output)}
            )
        return output
    
    def generate_code_constraints(self, stage_num=None, rekep_program_dir=None, seed=None, override=False):
        prompt = self._build_prompt_cost_functions(stage_num=stage_num)
        self.query_history.append(prompt)
        if stage_num is None:
            output_raw_file = os.path.join(rekep_program_dir, "output_raw.txt")
        else:
            output_raw_file = os.path.join(rekep_program_dir, "output_cost_function_stage_{}.txt".format(stage_num))
        if not os.path.exists(output_raw_file) or override:
            # stream back the response
            stream = query_vlm_model(self.client, self.vlm_model, self.query_history, TEMPERATURE, TOP_P, stream=True)
            output = ""
            start = time.time()
            for chunk in stream:
                print(f'[{time.time()-start:.2f}s] Querying OpenAI API...', end='\r')
                if chunk.choices[0].delta.content is not None:
                    output += chunk.choices[0].delta.content
            print(f'[{time.time()-start:.2f}s] Querying OpenAI API...Done')
            self.query_history.append(
                {"role": "system", "content": "{}".format(output)}
            )
            self.query_history.append(output)
            # save raw output
            with open(output_raw_file, 'w') as f:
                f.write(output)
        else:
            with open(output_raw_file, "r") as f:
                output = f.read()
        return output
    
    def generate(self, img, instruction, rekep_program_dir=None, hint="", seed=None, ):
        """
        Args:
            img (np.ndarray): image of the scene (H, W, 3) uint8
            instruction (str): instruction for the query
        Returns:
            save_dir (str): directory where the constraints
        """
        if rekep_program_dir is None:
            # create a directory for the task
            fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + instruction.lower().replace(" ", "_")
            self.task_dir = os.path.join(self.base_dir, fname)
            os.makedirs(self.task_dir, exist_ok=True)
            rekep_program_dir = self.task_dir
        self.task_dir = rekep_program_dir
        image_path = os.path.join(rekep_program_dir, 'query_img.png')
        cv2.imwrite(image_path, img[..., ::-1])
        if self.query_history is None:
            self.query_history = [{"role": "system", "content": "You are a helpful assistant that breaks down task, write geometric constraints, and write Python code for robot manipulation. Please learn from the knowledge as much as possible. Think carefully and reason step by step."}]
        
        self.generate_geometry_constraints(image_path, instruction, rekep_program_dir, hint, seed)
        output = self.generate_code_constraints(rekep_program_dir=rekep_program_dir)
        
        # parse and save constraints
        _parse_and_save_constraints(output, self.task_dir)
        return self.task_dir
