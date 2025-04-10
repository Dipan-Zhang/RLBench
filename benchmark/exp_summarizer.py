import numpy as np
import pickle
import os
def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def save_pickle(pickle_file, data):
    with open(pickle_file, 'wb') as pfile:
        pickle.dump(data, pfile)

import argparse
import pandas as pd
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trial', type=str, default='outputs/OpenMicrowave/ours/trial_2025-04-06_20-20/exp_results/exp_results_all.pkl')
    args = parser.parse_args()
    exp_results_fp = os.path.join(args.trial, 'exp_results', 'exp_results_all.pkl')
    exp_results = load_pickle(exp_results_fp)
    exp_summarized = {}
    for camera_name in exp_results.keys():
        exp_result = exp_results[camera_name]
        exp_summarized[camera_name] = np.mean(exp_result, axis=0)
        exp_results[camera_name] = exp_result
    exp_summarized['all'] = np.mean(list(exp_summarized.values()), axis=0)
    print(exp_summarized)

    # save summarized results
    exp_results_save_fp = os.path.join(args.trial, 'exp_results', 'summarized.csv')
    exp_results_df = pd.DataFrame(exp_results)
    exp_results_df.to_csv(exp_results_save_fp)
    print(f"Saved summarized results to {exp_results_save_fp}")