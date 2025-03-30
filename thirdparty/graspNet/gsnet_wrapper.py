
from thirdparty.graspNet.gsnet import GSNet, grasp_to_pointcloud, vis_save_grasp, get_best_grasp, get_pose_from_grasp, get_closest_grasp, get_default_grasp
from thirdparty.graspNet.utils.grasp_utils import crop_points
import open3d as o3d
import torch
import numpy as np
import traceback
from PIL import Image
import ipdb
import matplotlib.pyplot as plt
import os
import yaml

def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        # Load the YAML file into a Python dictionary
        config = yaml.safe_load(file)
    return config

class GSNetWrapper():
    """Wrapper for GSNet created by AR, move to my repo later"""
    def __init__(
            self, 
            cfgs=None,
        ):
        self.cfgs = cfgs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cam_w = cfgs['cam_w']
        self.cam_h = cfgs['cam_h']
        if self.cfgs["USE_GSNET"]:
            self.gsnet = GSNet(self.cfgs["gsnet"])
            
    def inference_gsnet(self, pcs, keep=1e6, nms=True):
        gg = self.gsnet.inference(pcs)
        if nms:
            gg = gg.nms()
        gg = gg.sort_by_score()
        if len(gg) > keep:
            gg = gg[:keep]
        if self.cfgs["gsnet"]["vis"]:
            grippers = gg.to_open3d_geometry_list()
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pcs.astype(np.float32))
            o3d.visualization.draw_geometries([cloud, *grippers]) 
            
        return gg
    

    def detect_grasp_gsnet(self, points, colors=None, save_vis=False):
        '''GSNet'''
        # need to preprocess point cloud
        pcs_input = points.copy()
        # pcs_input[..., 2] = -pcs_input[..., 2]  # flip z axis
        gg = self.inference_gsnet(pcs_input, nms=False)
        print(gg[0])
        #!!!!     # why such modification???
        # adjust grasps
        # for g_i in range(len(gg)):
        #     translation = gg[g_i].translation
        #     rotation = gg[g_i].rotation_matrix
        #     translation = np.array([translation[0], translation[1], -translation[2]])
        #     rotation[2, :] = -rotation[2, :]
        #     rotation[:, 2] = -rotation[:, 2]
        #     gg.grasp_group_array[g_i][13:16] = translation
        #     gg.grasp_group_array[g_i][4:13] = rotation.reshape(-1)
        # print(gg[0])
        print('grasp num:', len(gg))
        if save_vis:
            vis_save_grasp(points, gg, f"{self.cfgs['SAVE_ROOT']}/gsnet.ply")
        return gg

    def infer_best_grasp(self, pcd, position, max_dis=0.05, top_down=False):
        """Get best grasp from GSNet"""
        partial_points = np.array(pcd.points)
        partial_colors = np.array(pcd.colors)
        
        # visualization
        ds_points, _, _ = crop_points(position, partial_points, thres=0.5) # search points within cube (r=thres[m])
        save_pcd = o3d.geometry.PointCloud()
        save_pcd.points = o3d.utility.Vector3dVector(ds_points)
        save_pcd.colors = o3d.utility.Vector3dVector(np.array([1, 0, 0]) * np.ones((ds_points.shape[0], 3)))
        save_pcd.points.append(position)
        save_pcd.colors.append(np.array([0, 1, 0]))
        o3d.io.write_point_cloud(f"{self.cfgs['SAVE_ROOT']}/grasp_point.ply", save_pcd)
        
        MAX_ATTEMPTS = 20 # in case there is no good grasp at one time
        max_dis = max_dis # 
        best_grasp = None
        max_radius, min_radius = 0.2, 0.1
        gg = None # grasp group

        for num_attempt in range(MAX_ATTEMPTS):
            try:
                # generate grasp group (gg) around within crop radius
                crop_radius = max_radius - (max_radius - min_radius) * num_attempt / MAX_ATTEMPTS
                print('=> crop_radius:', crop_radius)
                cropped_points, cropped_colors, cropped_normals = crop_points(
                    position, partial_points, partial_colors, thres=crop_radius, save_root=self.cfgs['SAVE_ROOT']
                )
                try:
                    gg = self.detect_grasp_gsnet(cropped_points, cropped_colors, save_vis=True)
                except KeyboardInterrupt:
                    exit(0)
                except:
                    traceback.print_exc()
                if gg is None or len(gg) == 0:
                    continue
                print('=> total grasp:', len(gg))
                
                # select best grasp around selected pixel
                best_grasp = get_best_grasp(gg, position, max_dis=max_dis, top_down=False) # original: 0.03
                if best_grasp is not None:
                    break
                else:
                    print('==>> no best grasp')
            except KeyboardInterrupt:
                exit(0)
            except:
                traceback.print_exc()

        # if still no best -> use closest grasp
        if best_grasp is None:
            try:
                gg = self.detect_grasp_gsnet(cropped_points, cropped_colors, False)
            except:
                gg = self.detect_grasp_gsnet(partial_points, partial_colors, False)
            best_grasp = get_closest_grasp(gg, position)
            print('==>> use GSNet for closest grasp')
        vis_save_grasp(cropped_points, best_grasp, f"{self.cfgs['SAVE_ROOT']}/best_grasp.ply")
        return best_grasp
    
def pick_points_in_viewer(points, scene_colors=None, verbose=False) -> np.ndarray:
    def pick_points(pcd):
        print("")
        print(
            "1) Please pick at least three correspondences using [shift + left click]"
        )
        print("   Press [shift + right click] to undo point picking")
        print("2) After picking points, press 'Q' to close the window")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print("")
        return vis.get_picked_points()

    if isinstance(points, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if scene_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(scene_colors)
    elif isinstance(points, o3d.cuda.pybind.geometry.TriangleMesh):
        pcd = o3d.geometry.PointCloud()
        pcd.points = points.vertices
        pcd.colors = points.vertex_colors
    else:
        pcd = points

    picked_ids = pick_points(pcd)
    final_points = np.asarray(pcd.points)[picked_ids]
    print("Points selected: ", final_points.shape, final_points)

    if verbose:
        print("Final points: ")
        for i in range(len(final_points)):
            print(final_points[i])
    return final_points


if __name__ == '__main__':
    # cfgs = read_yaml_config(f"../RAM_code/run_realworld/configs/drawer_open.yaml")
    cfgs = read_yaml_config(f"./cfgs/simulation/config.yaml")
    os.makedirs(cfgs['SAVE_ROOT'], exist_ok=True)
    gsnet = GSNetWrapper(cfgs)

    pcd = o3d.io.read_point_cloud("../RAM_code/run_realworld/real_data/input/pcd.ply")
    rgb = Image.open("../RAM_code/run_realworld/real_data/input/rgb.png")
    
    pts = pick_points_in_viewer(pcd)
    ipdb.set_trace()
    # ret_dict = gym.lift_affordance(rgb, pcd, pixel.reshape(2,), post_contact_dir)
    best_grasp = gsnet.infer_best_grasp(pcd, pts[0], max_dis=0.06)

    print(best_grasp)
    ipdb.set_trace()
