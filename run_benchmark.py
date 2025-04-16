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
    parser.add_argument('--trial_name', type=str, help='batch eval using same trial name')
    parser.add_argument('--headless', action='store_true', help='run in headless mode')
    parser.add_argument('--summarize', action='store_true', help='run summarizer')
    args = parser.parse_args()

    method = args.method
    sim_cfg_fp = args.sim_config_fp
    sim_cfg = OmegaConf.load(sim_cfg_fp)
    TASK_LIST_PORTABLE = sim_cfg['PORTABLE_TASK_LIST']
    TASK_LIST_ARTICULATE = sim_cfg['ARTICULATE_TASK_LIST']

    if args.task == 'all':
        TASKS = TASK_LIST_PORTABLE + TASK_LIST_ARTICULATE
    elif args.task == 'portable':
        TASKS = TASK_LIST_PORTABLE
    elif args.task == 'articulate':
        TASKS = TASK_LIST_ARTICULATE
    else:
        TASKS = [args.task]

    for task in tqdm(TASKS):
        # print(f"====================Task: {task}=====================")
        # # task_env, task_name = task.split("@")
        taskName = underscore_string_to_camel_case(task)
        trial_name = os.path.join('./outputs', taskName, method, args.trial_name)
        cmd = f"python benchmark/test_affordance.py --task_name {task} --method {method} -t {trial_name}"
        if (
            args.headless
        ):  # and task_name not in ["OpenDishwasher", "CloseDishwasher"]:
            cmd += "--headless"
        os.system(cmd)

        if args.summarize:
            print(f"Running summarizer for {task}")
            cmd = f"python benchmark/exp_summarizer.py -t {trial_name}"
            os.system(cmd)

        print("=======================================================")
