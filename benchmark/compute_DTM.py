import numpy as np
import pickle
import os
import argparse
import pandas as pd
import cv2
import matplotlib.pyplot as plt

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
    "quick wrappper function to get a mask for a single frame using interactive annotation"
    # load images

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(f"please label the contact point")
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
        fig, ax = plt.subplots()
        ax.imshow(gt_mask)
        ax.scatter(corres[1], corres[0], color='red', marker='*', s=200, edgecolor='white', linewidth=1.25)
        save_DTM_fp = os.path.join(save_fp, 'DTM.png')
        fig.savefig(save_DTM_fp)
        plt.close(fig)
    return DTM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_dir', type=str, default='outputs/OpenMicrowave/ours/trial_2025-04-06_20-20/')
    args = parser.parse_args()
    

    base_fp = '/'.join(args.trial.split('/')[:-2])
    obs_fp = os.path.join(base_fp, 'obs')
    transferred_results_fp = os.path.join(args.trial, 'transferred_motion_all.pkl')
    transferred_results = load_pickle(transferred_results_fp)

    # load DTM GT if exists
    DTM_all = {}
    camera_names = os.listdir(obs_fp)
    for camera_name in camera_names:
        DTM_per_cam = []
        rgb_fn = os.path.join(obs_fp, camera_name, 'rgb.png')
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
            rgb_cropped = cv2.imread(rgb_cropped_fn, -1)[..., [2,1,0]]

            contact_pt = interactive_mask(rgb_cropped)
            mask_gt = np.zeros(rgb_cropped.shape[:2], dtype=np.uint8)
            mask_gt[int(contact_pt[0][1]), int(contact_pt[0][0])] = 1
            np.save(mask_gt_fp, mask_gt)
        
        for idx, trial in enumerate(transferred_results[camera_name]):
            corres_mask = transferred_results[camera_name][trial]['corres_mask']
            corres_pixel = np.array(np.where(corres_mask > 0))
            save_dir = os.path.join(args.trial, camera_name, 'trial_{}'.format(idx))
            DTM = compute_DTM(mask_gt, corres_pixel, 20, save_dir)
            DTM_normalized = DTM / diagonal
            DTM_per_cam.append(DTM_normalized)

        DTM_all[camera_name] = np.mean(DTM_per_cam, axis=0)

    print(DTM_all)
    DTM_results_save_fp = os.path.join(args.trial, 'exp_results')
    os.makedirs(DTM_results_save_fp, exist_ok=True)

    DTM_results_save_fn = os.path.join(DTM_results_save_fp, 'DTM_results.csv')
    DTM_results_df = pd.DataFrame(DTM_all)
    DTM_results_df.to_csv(DTM_results_save_fn)
    print(f"Saved DTM results to {DTM_results_save_fn}")
