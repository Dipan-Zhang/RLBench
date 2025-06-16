import numpy as np
import os
from tqdm import tqdm
import argparse
from benchmark.helpers import underscore_string_to_camel_case
from omegaconf import OmegaConf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='all', help='task name')
    parser.add_argument('--method', type=str, default='ours', help='method name' )
    parser.add_argument('--sim_config_fp', type=str, default='./cfgs/config.yaml', help='config file path')
    parser.add_argument('--trial_dir', type=str, help='batch eval using same trial directory')
    parser.add_argument('--headless', action='store_true', help='run in headless mode')
    parser.add_argument('--DEBUG_VIS', action='store_true', help='visualize the motion in rlbench')
    parser.add_argument('--num_var', type=int, default=0, help='use variation')

    args = parser.parse_args()

    method = args.method
    sim_cfg_fp = args.sim_config_fp
    sim_cfg = OmegaConf.load(sim_cfg_fp)
    TASK_LIST_PORTABLE = sim_cfg['PORTABLE_TASK_LIST']
    TASK_LIST_ARTICULATE = sim_cfg['ARTICULATE_TASK_LIST']

    if args.task == 'all':
        TASKS =  TASK_LIST_ARTICULATE + TASK_LIST_PORTABLE
    elif args.task == 'portable':
        TASKS = TASK_LIST_PORTABLE
    elif args.task == 'articulate':
        TASKS = TASK_LIST_ARTICULATE
    elif args.task == 'flex':
        TASKS = ['down_toilet_seat', 'close_laptop']
    elif args.task == 'bechmarkv1_rest':
        TASKS = ['open_drawer', 'close_microwave', 'open_slide_cabinet', 'close_slide_cabinet', 'close_cabinet']
    elif args.task == 'ablation_multiple_goal_rest':
        TASKS = ['open_drawer', 'open_dishwasher', 'down_toilet_seat', 'close_slide_cabinet', 'open_slide_cabinet']
        
    elif args.task =='ablation_2D':
        TASKS = [
            'open_microwave', 
            'close_microwave',
            'open_drawer', 
            'open_dishwasher',
            'open_slide_cabinet',
            'close_laptop',
            'open_cabinet',
        ]
    else:
        TASKS = [args.task]

    if args.task == 'ablation_var':
        SAVE_BASE_DIR = '../RLBench/outputs_ablation_var'
    elif args.task == 'ablation_2D':
        SAVE_BASE_DIR = '../RLBench/outputs_ablation_2D'
    else:
        SAVE_BASE_DIR = '../RLBench/outputs'

    for task in tqdm(TASKS):
        print(f"====================Task: {task}=====================")
        taskName = underscore_string_to_camel_case(task)
        if args.num_var == 0:
            trial_dir = os.path.join(SAVE_BASE_DIR, taskName, method, args.trial_dir)
            cmd = f"python benchmark/test_affordance.py --task_name {task} --method {method} --trial_dir {trial_dir}"
            if args.headless:
                cmd += " --headless"
            if args.DEBUG_VIS:
                cmd += " --DEBUG_VIS"
            os.system(cmd)
        else:
            for var in range(args.num_var):
                trial_dir = os.path.join(SAVE_BASE_DIR, taskName, method, args.trial_dir, f'var_{var}')
                cmd = f"python benchmark/test_affordance.py --task_name {task} --method {method} --trial_dir {trial_dir}"
                if args.headless:
                    cmd += " --headless"
                if args.DEBUG_VIS:
                    cmd += " --DEBUG_VIS"
                os.system(cmd)

        print("=======================================================")
