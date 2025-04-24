import numpy as np
import os
from tqdm import tqdm
import argparse
from benchmark.helpers import underscore_string_to_camel_case
from omegaconf import OmegaConf
import shutil
"move the trajectory files from hz to my convention"

def hz_to_rlbench_cam(camera_name):
    hz_cam_convert = {
        'cam_overhead': 'default',
        'cam_front': 'default',
        'cam_over_shoulder_left': 'left',
        'cam_over_shoulder_right': 'right',
    }
    return hz_cam_convert.get(camera_name, camera_name)

def rlbench_to_hz_cam(camera_name):
    rlbench_cam_convert = {
        'default': 'cam_overhead',
        'left': 'cam_over_shoulder_left',
        'right': 'cam_over_shoulder_right',
    }
    return rlbench_cam_convert.get(camera_name, camera_name)

def get_time():
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-t', '--task', type=str, default='all', help='task name')
    parser.add_argument('-i', '--input_dir', type=str, default='./rlbench_anran/', help='input directory')
    args = parser.parse_args()


    input_dir = args.input_dir
    sim_cfg_fp = './cfgs/config.yaml'
    sim_cfg = OmegaConf.load(sim_cfg_fp)
    TASK_LIST_PORTABLE = sim_cfg['PORTABLE_TASK_LIST']
    TASK_LIST_ARTICULATE = sim_cfg['ARTICULATE_TASK_LIST']
    TASKS = TASK_LIST_PORTABLE + TASK_LIST_ARTICULATE


    trial_name = f'trial_{get_time()}'
    hz_cams = ['default', 'left', 'right']
    methods = ['gflow', 'vrb', 'where2act']
    for task in tqdm(TASKS):
        # print(f"====================Task: {task}=====================")
        # # task_env, task_name = task.split("@")
        taskName = underscore_string_to_camel_case(task)
        for cam_name in hz_cams:
            for method in methods:
                save_path = os.path.join(input_dir, taskName, f'{cam_name}')
                file_path = os.path.join(save_path, f'traj_{method}_000000.npz')
                if not os.path.exists(file_path):
                    print(f"File {file_path} does not exist, skipping...")
                    continue

                bench_cam = rlbench_to_hz_cam(cam_name)
                trial_dir = os.path.join('./outputs', taskName, method, trial_name, bench_cam)
                os.makedirs(trial_dir, exist_ok=True)
                save_file_path = os.path.join(trial_dir, f'traj_{method}_000000.npz')
                shutil.copyfile(file_path, save_file_path)
                print(f"Copying {file_path} to {save_file_path}")
        
        print("=======================================================")
