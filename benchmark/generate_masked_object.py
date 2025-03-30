import cv2 
import numpy as np  
import os
import matplotlib.pyplot as plt
import argparse
import json

# from affordance.helpers import backproject, visualize_points
"""
TO BE DEPRECATED => use interative mask instead
Only use it for image that you want to reconstruct with NEUS, because it will resize the image
generate and resize the cropped image of the masked object -> for kinect dataset
"""
#TODO: integrate this with mask generation

def compute_box_from_mask(mask):
    idxs = np.where(mask)
    x1, x2 = np.amin(idxs[1]), np.amax(idxs[1])
    y1, y2 = np.amin(idxs[0]), np.amax(idxs[0])
    return int(x1), int(y1), int(x2), int(y2)


def compute_cropped_intrinsics(cam_K, resize, crop_center, res):
    # This implementation is tested faithfully. Results in PnP with 0.02% drop.
    K = cam_K.copy()

    # First resize from original size to the target size
    K[0, 0] = K[0, 0] * resize
    K[1, 1] = K[1, 1] * resize
    K[0, 2] = (K[0, 2] + 0.5) * resize - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * resize - 0.5

    # Then crop the image --> need to modify the optical center,
    # remember that current top left is the coordinates measured in resized results
    # And its information is vu instead of uv
    top_left = crop_center * resize - res / 2
    K[0, 2] = K[0, 2] - top_left[1]
    K[1, 2] = K[1, 2] - top_left[0]
    return K

def crop_and_pad_image(
    img,
    center,
    scale,
    res=None,
    channel=3,
    interpolation=cv2.INTER_LINEAR,
    resize=True,
):
    # Code from CDPN
    ht, wd = img.shape[0], img.shape[1]
    dtype = img.dtype
    upper = max(0, int(center[0] - scale / 2.0 + 0.5))
    left = max(0, int(center[1] - scale / 2.0 + 0.5))
    bottom = min(ht, int(center[0] - scale / 2.0 + 0.5) + int(scale))
    right = min(wd, int(center[1] - scale / 2.0 + 0.5) + int(scale))
    crop_ht = float(bottom - upper)
    crop_wd = float(right - left)

    # resize to preset resolution
    if resize:
        if crop_ht > crop_wd:
            resize_ht = res
            resize_wd = int(res / crop_ht * crop_wd + 0.5)
        elif crop_ht < crop_wd:
            resize_wd = res
            resize_ht = int(res / crop_wd * crop_ht + 0.5)
        else:
            resize_wd = resize_ht = int(res)
    if channel <= 3:
        tmpImg = img[upper:bottom, left:right]
        if not resize:
            if channel == 3:
                outImg = np.ones((int(scale), int(scale), channel), dtype=dtype) * 0.5
            else:
                outImg = np.zeros((int(scale), int(scale), channel), dtype=dtype)
            outImg[
                int(scale / 2.0 - (bottom - upper) / 2.0 + 0.5) : (
                    int(scale / 2.0 - (bottom - upper) / 2.0 + 0.5) + (bottom - upper)
                ),
                int(scale / 2.0 - (right - left) / 2.0 + 0.5) : (
                    int(scale / 2.0 - (right - left) / 2.0 + 0.5) + (right - left)
                ),
                :,
            ] = tmpImg
            return outImg
        resizeImg = cv2.resize(
            tmpImg, (resize_wd, resize_ht), interpolation=interpolation
        )
        # print(tmpImg.shape, scale)
        if len(resizeImg.shape) < 3:
            resizeImg = np.expand_dims(
                resizeImg, axis=-1
            )  # for depth image, add the third dimension
        if channel == 3:
            outImg = np.ones((int(res), int(res), channel), dtype=dtype) * 125
            outImg = outImg.astype(dtype)
        else:
            outImg = np.zeros((int(res), int(res), channel), dtype=dtype)
        outImg[
            int(res / 2.0 - resize_ht / 2.0 + 0.5) : (
                int(res / 2.0 - resize_ht / 2.0 + 0.5) + resize_ht
            ),
            int(res / 2.0 - resize_wd / 2.0 + 0.5) : (
                int(res / 2.0 - resize_wd / 2.0 + 0.5) + resize_wd
            ),
            :,
        ] = resizeImg

    else:
        raise NotImplementedError
    return outImg

def show_(img):
    plt.imshow(img)
    plt.show()

def read_image(color_init_path, depth_init_path, mask_init_path):
    color_init = cv2.imread(color_init_path)[..., ::-1]
    depth_init = cv2.imread(depth_init_path, cv2.IMREAD_UNCHANGED)
    mask_init = cv2.imread(mask_init_path, cv2.IMREAD_GRAYSCALE)
    mask_init = mask_init * 255

    return color_init, depth_init, mask_init

def remove_background(image):
    """
    Remove the background of the image and fill with white.
    Assumes input image is RGBA.
    """
    # Convert the image to a numpy array
    image_np = np.array(image)
    r, g, b, a = image_np[..., 0], image_np[..., 1], image_np[..., 2], image_np[..., 3]

    # Create a white background (RGB only)
    white_bg = np.ones_like(image_np[..., :3]) * 255

    # Use the alpha channel as the mask
    mask = a > 100
    image_rgb = np.where(mask[..., None], image_np[..., :3], white_bg)

    # Optional: If you want RGBA output, include the alpha channel
    image_rgba = np.concatenate([image_rgb, a[..., None]], axis=-1)

    # Debugging visualization
    plt.imshow(image_rgba.astype(np.uint8))
    plt.title("RGBA Image with Background Removed")
    plt.show()

    return image_rgba

def compute_cropping_params(mask, pad_ratio, resolution=256):
    bbox = np.array(compute_box_from_mask(mask))
    center = np.array([bbox[1] + bbox[3], bbox[0] + bbox[2]]) / 2  # [h, w]
    crop_scale = max(bbox[3] - bbox[1], bbox[2] - bbox[0]) * pad_ratio
    resize_ratio = resolution / crop_scale
    return center, crop_scale, resize_ratio

def crop_images(color, mask, depth, center, crop_scale, resolution):
    color_cropped = crop_and_pad_image(color, center, crop_scale, resolution, channel=3)
    mask_cropped = crop_and_pad_image(mask, center, crop_scale, resolution, channel=1, interpolation=cv2.INTER_NEAREST)
    depth_cropped = crop_and_pad_image(depth, center, crop_scale, resolution, channel=1, interpolation=cv2.INTER_NEAREST)
    return color_cropped, mask_cropped, np.squeeze(depth_cropped, axis=2)

def save_images(data_path, object_name, idx, rgba, depth, mask, single=False):
    if single:
        cv2.imwrite(os.path.join(data_path, f'{object_name}.png'), rgba)
        cv2.imwrite(os.path.join(data_path, f'{object_name}_depth.png'), depth.astype(np.uint16))
        cv2.imwrite(os.path.join(data_path, f'{object_name}_mask.png'), mask)
    else:
        cv2.imwrite(os.path.join(data_path, f'{object_name}_{idx}.png'), rgba)
        cv2.imwrite(os.path.join(data_path, f'{object_name}_depth_{idx}.png'), depth.astype(np.uint16))
        cv2.imwrite(os.path.join(data_path, f'{object_name}_mask_{idx}.png'), mask)


def process(data_path, color_init, mask_init, depth_init, object_name, cam_k, idx=0, SINGLE=False, rm_bg=True):
    center, crop_scale, resize_ratio = compute_cropping_params(mask_init, pad_ratio=1.25)
    cam_k_resized = compute_cropped_intrinsics(cam_k, resize_ratio, center, 256)
    
    object_color, object_mask, object_depth = crop_images(color_init, mask_init, depth_init, center, crop_scale, 256)

    if rm_bg:
        object_rgba = remove_background(np.concatenate([object_color, object_mask], axis=-1))
    else:
        object_rgba = np.concatenate([object_color, object_mask], axis=-1)

    object_rgba = object_rgba[..., [2, 1, 0, 3]]  # RGB -> BGR
    save_images(data_path, object_name, object_name, idx, object_rgba, object_depth, object_mask, SINGLE)

    return cam_k_resized


"""example
python ./dataset/generate_masked_object.py --object_name kitchen_knife --object_name knife_0 --frame_idx 40
python ./dataset/generate_masked_object.py --object_name hand_held --object_name cup_0 --frame_idx 03
"""

if __name__ == '__main__':
    "process each object in the scene to single cropped object based the mask "
    parser = argparse.ArgumentParser(description='crop masked object images')
    parser.add_argument('--object_name', type=str, help='Name of the scene')
    parser.add_argument('--object_name', type=str, help='Name of the dataset')
    parser.add_argument('--num', type=int, help='number of objects, for scene with multiple objects')
    parser.add_argument('--frame_idx', type=str, help='frame index of rgb/depth image')
    args = parser.parse_args()

    object_name = args.object_name
    object_name = args.object_name
    num = args.num if args.num else 0
    frame_idx = args.frame_idx

    # scene with single object
    data_path = f'./dataset/kinect/{object_name}/'
    color_init_path = os.path.join(data_path, 'color/{:6d}.jpg'.format(frame_idx))
    depth_init_path = os.path.join(data_path, 'depth/{:6d}.png'.format(frame_idx))
    mask_init_path = os.path.join(data_path, f'mask_{frame_idx}.png')
    color_init, depth_init, mask_init = read_image(color_init_path, depth_init_path, mask_init_path)

    intrinsic_fname = os.path.join(data_path, 'camera_intrinsic.json')
    with open(intrinsic_fname, 'r') as f:
        intrinsic = json.load(f)
    cam_k = np.array(intrinsic['intrinsic_matrix']).reshape(3, 3).T

    cam_k_resized = process(data_path, color_init, mask_init, depth_init, object_name, cam_k, num, SINGLE=True, rm_bg=True)
    # write resized intrinsic matrix to json file
    intrinsic['intrinsic_resized'] = cam_k_resized.T.flatten().tolist()
    with open(intrinsic_fname, 'w') as f:
        json.dump(intrinsic, f)


    # # TEST correctness of new intrinsic matrix
    # # import ipdb; ipdb.set_trace()
    # import open3d as o3d
    # object_pt, _ = backproject(depth_init, cam_k, mask_init, False)
    # object_pcd = visualize_points(object_pt)

    # cropped_object_pt, _ = backproject(object_depth_init, cam_k_resized, object_depth_init>0, False)
    # cropped_object_pcd = visualize_points(cropped_object_pt)
    # cropped_object_pcd.paint_uniform_color([0, 1, 0])   

    # o3d.visualization.draw_geometries([object_pcd, cropped_object_pcd])


    # scene with multiple objects: for loop to iterate through all objects
    