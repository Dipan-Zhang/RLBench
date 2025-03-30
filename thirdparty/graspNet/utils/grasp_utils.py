import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def crop_points(point, pcd_points, pcd_colors=None, thres=0.2, save_root=None):
    '''crop pcd close to a given point'''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    if pcd_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    [k, idx, _] = pcd_tree.search_radius_vector_3d(point, thres)
    pcd = pcd.select_by_index(idx)
    if save_root:
        o3d.io.write_point_cloud(f"{save_root}/cropped_pcd.ply", pcd)
    ret_points = np.array(pcd.points)
    ret_colors = np.array(pcd.colors) if pcd_colors is not None else None
    ret_normals = np.array(pcd.normals)
    return ret_points, ret_colors, ret_normals

def save_frames_as_gif(frames, path='./', filename=f'gifs/re3_multitask.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=30)
