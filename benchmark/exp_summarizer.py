import numpy as np
import pickle
import os
import argparse
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

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

def interactive_mask(img):
    "quick wrapper function to get a mask for a single frame using interactive annotation"
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(f"Please label the contact point")
    ax.imshow(img)

    points = []
    labels = []

    def onclick(event):
        if event.button == 1:  # Left click for positive point
            points.append([event.xdata, event.ydata])
            labels.append(1)
            ax.scatter(event.xdata, event.ydata, color='green', marker='*', s=200, edgecolor='white', linewidth=1.25)
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return points

def compute_DTM(gt_mask, corres, contour_sz, save_fp=None):
    """
    Compute the Dense Tracking Map (DTM) from the ground truth mask and correspondences.
    :param gt_mask: Ground truth mask
    :param corres: Correspondences (1,2)
    :param contour_sz: Size of the contour
    :return: DTM
    """
    center_gt = np.array(np.where(gt_mask > 0))
    dist = np.linalg.norm(center_gt - corres, axis=0)
    if dist < contour_sz:
        DTM = 0
    else:
        DTM = dist - contour_sz
    
    if save_fp is not None:
        os.makedirs(save_fp, exist_ok=True)
        fig, ax = plt.subplots()
        ax.imshow(gt_mask)
        ax.scatter(corres[1], corres[0], color='red', marker='*', s=200, edgecolor='white', linewidth=1.25)
        save_DTM_fp = os.path.join(save_fp, 'DTM.png')
        fig.savefig(save_DTM_fp)
        plt.close(fig)
    return DTM

def calculate_dtm_metrics(trial_path):
    """Calculate DTM metrics for the given trial"""
    base_fp = '/'.join(trial_path.split('/')[:-2])
    obs_fp = os.path.join(base_fp, 'obs')
    transferred_results_fp = os.path.join(trial_path, 'transferred_motion_all.pkl')
    
    if not os.path.exists(transferred_results_fp):
        print(f"Warning: No transferred motion data found at {transferred_results_fp}")
        return {}
        
    transferred_results = load_pickle(transferred_results_fp)

    # load DTM GT if exists
    DTM_all = {}
    camera_names = os.listdir(obs_fp)
    
    for camera_name in camera_names:
        DTM_per_cam = []
        rgb_fn = os.path.join(obs_fp, camera_name, 'rgb.png')
        
        if not os.path.exists(rgb_fn):
            print(f"Warning: RGB image not found at {rgb_fn}, skipping {camera_name}")
            continue
            
        rgb = cv2.imread(rgb_fn, -1)[..., [2,1,0]]
        diagonal = np.linalg.norm(rgb.shape[:2])

        # load gt mask
        mask_gt_fp = os.path.join(obs_fp, camera_name, 'mask_gt.npy')
        if os.path.exists(mask_gt_fp):
            mask_gt = np.load(mask_gt_fp)
            mask_gt = mask_gt.astype(np.uint8)
        else:
            # label gt in place
            rgb_cropped_fn = os.path.join(obs_fp, camera_name, 'cropped_color.png')
            
            if not os.path.exists(rgb_cropped_fn):
                print(f"Warning: Cropped RGB not found at {rgb_cropped_fn}, skipping DTM for {camera_name}")
                continue
                
            rgb_cropped = cv2.imread(rgb_cropped_fn, -1)[..., [2,1,0]]
            print(f"\nPlease annotate the contact point for {camera_name}")
            contact_pt = interactive_mask(rgb_cropped)
            
            if not contact_pt:
                print(f"No point selected for {camera_name}, skipping")
                continue
                
            mask_gt = np.zeros(rgb_cropped.shape[:2], dtype=np.uint8)
            mask_gt[int(contact_pt[0][1]), int(contact_pt[0][0])] = 1
            np.save(mask_gt_fp, mask_gt)
        
        if camera_name not in transferred_results:
            print(f"Warning: No transfer results for {camera_name}")
            continue
            
        for idx, trial in enumerate(transferred_results[camera_name]):
            if 'corres_mask' not in transferred_results[camera_name][trial]:
                print(f"Warning: No correspondence mask for {camera_name}, trial {trial}")
                continue
                
            corres_mask = transferred_results[camera_name][trial]['corres_mask']
            if np.sum(corres_mask) == 0:
                print(f"Warning: Empty correspondence mask for {camera_name}, trial {trial}")
                continue
                
            corres_pixel = np.array(np.where(corres_mask > 0))
            save_dir = os.path.join(trial_path, camera_name, f'trial_{idx}')
            DTM = compute_DTM(mask_gt, corres_pixel, 20, save_dir)
            DTM_normalized = DTM / diagonal
            DTM_per_cam.append(DTM_normalized)

        if DTM_per_cam:
            DTM_all[camera_name] = DTM_per_cam
    
    return DTM_all

def load_success_rates(trial_path):
    """Load success rates from experiment results"""
    exp_results_fp = os.path.join(trial_path, 'exp_results', 'exp_results_all.pkl')
    
    if not os.path.exists(exp_results_fp):
        print(f"Warning: No experiment results found at {exp_results_fp}")
        return {}
        
    exp_results = load_pickle(exp_results_fp)
    success_rates = {}
    
    for camera_name in exp_results.keys():
        success_rates[camera_name] = exp_results[camera_name]
        
    return success_rates

def combine_metrics_report(trial_path, dtm_data, success_data):
    """Combine DTM and success rate metrics into a single report"""
    # Create output directory
    results_dir = os.path.join(trial_path, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare combined report
    combined_report = {}
    all_cameras = set(list(dtm_data.keys()) + list(success_data.keys()))
    
    for camera in all_cameras:
        camera_report = {'DTM': {}, 'Success': {}}
        
        # Process DTM data
        if camera in dtm_data and dtm_data[camera]:
            dtm_values = dtm_data[camera]
            camera_report['DTM']['values'] = dtm_values
            camera_report['DTM']['mean'] = np.mean(dtm_values)
            camera_report['DTM']['std'] = np.std(dtm_values)
        
        # Process success rate data
        if camera in success_data and len(success_data[camera]) > 0:
            success_values = success_data[camera]
            camera_report['Success']['values'] = success_values
            camera_report['Success']['mean'] = np.mean(success_values)
            camera_report['Success']['std'] = np.std(success_values)
        
        combined_report[camera] = camera_report
    
    # Calculate aggregate metrics across all cameras
    all_dtms = []
    all_successes = []
    
    for camera, data in combined_report.items():
        if 'DTM' in data and 'values' in data['DTM']:
            all_dtms.extend(data['DTM']['values'])
        if 'Success' in data and 'values' in data['Success']:
            all_successes.extend(data['Success']['values'])
    
    combined_report['all_cameras'] = {
        'DTM': {
            'mean': np.mean(all_dtms) if all_dtms else None,
            'std': np.std(all_dtms) if all_dtms else None
        },
        'Success': {
            'mean': np.mean(all_successes) if all_successes else None, 
            'std': np.std(all_successes) if all_successes else None
        }
    }
    
    # Save full results
    save_pickle(os.path.join(results_dir, 'combined_metrics.pkl'), combined_report)
    
    # Create summary dataframe
    summary_data = []
    for camera in combined_report:
        row = {'Camera': camera}
        
        if camera != 'all_cameras':
            if 'DTM' in combined_report[camera] and 'mean' in combined_report[camera]['DTM']:
                row['DTM_mean'] = combined_report[camera]['DTM']['mean']
                row['DTM_std'] = combined_report[camera]['DTM']['std']
            
            if 'Success' in combined_report[camera] and 'mean' in combined_report[camera]['Success']:
                row['Success_rate'] = combined_report[camera]['Success']['mean']
                row['Success_std'] = combined_report[camera]['Success']['std']
        else:
            # Handle the aggregated metrics
            if 'DTM' in combined_report[camera] and 'mean' in combined_report[camera]['DTM']:
                row['DTM_mean'] = combined_report[camera]['DTM']['mean']
                row['DTM_std'] = combined_report[camera]['DTM']['std']
            
            if 'Success' in combined_report[camera] and 'mean' in combined_report[camera]['Success']:
                row['Success_rate'] = combined_report[camera]['Success']['mean']
                row['Success_std'] = combined_report[camera]['Success']['std']
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(results_dir, 'metrics_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Print summary
    print("\n===== Performance Evaluation Summary =====")
    print(summary_df)
    print(f"\nFull results saved to {results_dir}")
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description="Evaluate task performance using DTM and success rates")
    parser.add_argument('--trial_dir', type=str, required=True,
                       help="Path to the trial directory (e.g., outputs/OpenMicrowave/ours/trial_2023-11-01_12-30)")
    parser.add_argument('--skip-dtm', action='store_true', help="Skip DTM calculation (use existing data)")
    args = parser.parse_args()
    
    # Validate trial path
    if not os.path.exists(args.trial):
        print(f"Error: Trial directory {args.trial} does not exist.")
        return
    
    print(f"Evaluating performance for trial: {args.trial}")
    
    # Calculate DTM if needed
    dtm_data = {}
    if not args.skip_dtm:
        print("Calculating DTM metrics...")
        dtm_data = calculate_dtm_metrics(args.trial)
    else:
        # Try to load existing DTM data
        dtm_path = os.path.join(args.trial, 'evaluation_results', 'combined_metrics.pkl')
        if os.path.exists(dtm_path):
            combined_data = load_pickle(dtm_path)
            dtm_data = {k: v.get('DTM', {}).get('values', []) for k, v in combined_data.items() 
                      if k != 'all_cameras' and 'DTM' in v}
            print("Loaded existing DTM data")
        else:
            print("Warning: No existing DTM data found and --skip-dtm specified")
    
    # Load success rates
    print("Loading success rates...")
    success_data = load_success_rates(args.trial)
    
    # Generate combined report
    print("Generating combined report...")
    combine_metrics_report(args.trial, dtm_data, success_data)

if __name__ == '__main__':
    main()