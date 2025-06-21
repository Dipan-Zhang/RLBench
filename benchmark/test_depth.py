import numpy as np

import os 
import cv2
import matplotlib.pyplot as plt
import json
import open3d as o3d

"test depth image backprojection and visualization for HZ inference"
def backproject(depth, intrinsics, instance_mask, NOCS_convention=True):
    """backproject depth image to 3d points
    Args:
        depth: [h, w]
        intrinsics: [3, 3]
        instance_mask: [h, w]
    return: pts: [num_pixel, 3], idxs: [2, num_pixel]
    """
    intrinsics_inv = np.linalg.inv(intrinsics)
    non_zero_mask = depth > 0
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)

    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0)  # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid  # [3, num_pixsel]
    xyz = np.transpose(xyz)  # [num_pixel, 3]

    z = depth[idxs[0], idxs[1]]

    pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
    if NOCS_convention:
        pts[:, 1] = -pts[:, 1]
        pts[:, 2] = -pts[:, 2]
    return pts, idxs


def visualize_points(points, colors=None):
    "take points and return open3d pcd"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


depth = cv2.imread('benchmark_dataset/CloseDrawer/default/depth_000000.png', cv2.IMREAD_UNCHANGED)
depth = depth.astype(np.float32) / 1000.0

camK = np.array([[626.42487143,   0.,         228.,        ],
            [  0.,         626.42487143, 128.,        ],
            [  0.,           0.,           1.,        ]])

mask = cv2.imread('benchmark_dataset/CloseDrawer/default/mask_000000.png', cv2.IMREAD_UNCHANGED)
mask = mask.astype(np.uint8)

pts,_ = backproject(depth, camK, mask, NOCS_convention=False)
pcd = visualize_points(pts, None) 

meta_data_fp = 'benchmark_dataset/CloseDrawer/default/meta_000000.json'
with open(meta_data_fp, 'r') as f:
    meta_data = json.load(f)


contact_point = np.array(meta_data['contact_point'])
T_cam_gripper = np.array(meta_data['T_cam_gripper']).reshape(4, 4)

pt_vis = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
pt_vis.translate(contact_point)

# Create coordinate system for gripper
gripper_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
gripper_vis.transform(T_cam_gripper)

gripper_vis.paint_uniform_color([1, 0, 0])  # red
world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

# Apply the transformation matrix to align with gripper pose
o3d.visualization.draw_geometries([pcd, pt_vis, gripper_vis, world])
breakpoint()