#! /usr/bin/env python3

import open3d as o3d
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np
import yaml
import shutil

from bev_generator import BEVGenerator
from logger import get_logger


logger = get_logger("debug")


def project_pcd_to_camera(pcd_input, camera_matrix, image_size, rvec=None, tvec=None):
    ''' Project pointcloud to camera '''
    
    assert rvec is not None, "rvec must be provided"
    assert tvec is not None, "tvec must be provided"
    
    points = pcd_input.point['positions'].numpy()
    colors = pcd_input.point['colors'].numpy()  # Extract 
    
    projected_points, _ = cv2.projectPoints(points, rvec=rvec, tvec=tvec, 
                                             cameraMatrix=camera_matrix, distCoeffs=None)

    projected_points = projected_points.reshape(-1, 2)
    
    img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)  # Create a blank image

    for idx, (point, color) in enumerate(zip(projected_points, colors)):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            color_bgr = (int(color[2]), int(color[1]), int(color[0]))
            cv2.circle(img, (x, y), 3, color_bgr, -1)
    
    return img


def plot_segmentation_classes(mask: np.ndarray, path: str = None, title: str = None) -> None:
    """
    Reads a single channel segmentation mask and plots (x,y) coordinates of each unique class.
    """
    unique_classes = np.unique(mask)
    
    plt.figure(figsize=(10, 6))
    
    # Load color map from config
    with open("config/Mavis.yaml", 'r') as f:
        config = yaml.safe_load(f)
    color_map = config['color_map']
    
    for class_id in unique_classes:
        y_coords, x_coords = np.where(mask == class_id)
        
        # Get color for the class, default to black if not found
        color = color_map.get(class_id, [0, 0, 0])
        color = [c/255 for c in color[::-1]] # Convert BGR to RGB and normalize
        
        plt.scatter(x_coords, y_coords, label=f'Class {class_id}', color=color, alpha=0.5)
    
    if title:
        plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)  # save plot to disk
    else:
        plt.show()
    plt.close()  # close the plot to free memory


if __name__ == "__main__":  
     
    
     

    # ================================================
    # CASE 9: testing dz consistency across consecutive rectified frames
    # ================================================

    # pcd_dir = f"debug/frames-2"
    # output_seg_masks_dir = f"debug/seg-masks-2"

    # os.makedirs(output_seg_masks_dir, exist_ok=True)
    
    # success_count = 0
    # total_count = 0

    # bev_generator = BEVGenerator()
    
    # pcd_files = []
    # for root, _, files in os.walk(pcd_dir):
    #     for file in files:
    #         if file == "left-segmented-labelled.ply":
    #             pcd_files.append(os.path.join(root, file))
    
    # total_count = len(pcd_files)
    
    # pcd_files.sort()
    
    # rectified_pcd_files = []

    # for pcd_path in tqdm(pcd_files, desc="Processing point clouds"):
    #     try:
    #         pcd_input = o3d.t.io.read_point_cloud(pcd_path)
            
    #         logger.warning(f"================================================")
    #         logger.warning(f"pcd_path: {pcd_path}")
    #         logger.warning(f"================================================\n")

    #         pcd_rectified = bev_generator.tilt_rectification(pcd_input)
    #         rectified_pcd_files.append(pcd_rectified)

    #         # original pointcloud
    #         x_i = pcd_input.point['positions'][:, 0].numpy()
    #         y_i = pcd_input.point['positions'][:, 1].numpy()
    #         z_i = pcd_input.point['positions'][:, 2].numpy()

    #         # rectified pointcloud
    #         x_r = pcd_rectified.point['positions'][:, 0].numpy()
    #         y_r = pcd_rectified.point['positions'][:, 1].numpy()
    #         z_r = pcd_rectified.point['positions'][:, 2].numpy()
            
    #         # logger.warning(f"================================================")
    #         # logger.warning(f"x_i ==> [{x_i.min():.2f}, {x_i.max():.2f}]")
    #         # logger.warning(f"y_i ==> [{y_i.min():.2f}, {y_i.max():.2f}]")
    #         # logger.warning(f"z_i ==> [{z_i.min():.2f}, {z_i.max():.2f}]")
    #         # logger.warning(f"================================================\n")
            

    #         logger.info(f"================================================")
    #         logger.info(f"x_r ==> [{x_r.min():.2f}, {x_r.max():.2f}] [mean: {(x_r.min() + x_r.max()) / 2:.2f}]")
    #         logger.info(f"y_r ==> [{y_r.min():.2f}, {y_r.max():.2f}] [mean: {(y_r.min() + y_r.max()) / 2:.2f}]")
    #         logger.info(f"z_r ==> [{z_r.min():.2f}, {z_r.max():.2f}] [mean: {(z_r.min() + z_r.max()) / 2:.2f}]")
    #         logger.info(f"================================================\n")

    #     except Exception as e:
    #         logger.error(f"Error processing {pcd_path}: {e}")

    # ================================================
    # CASE 8: testing compute_tilt_matrix success-rate
    # ================================================

    pcd_dir = f"debug/frames-2"
    
    seg_masks_dir = f"debug/2/seg-masks"
    rectified_pcd_dir = f"debug/2/rectified-pcd"
    left_img_dir = f"debug/2/left-imgs"
    right_img_dir = f"debug/2/right-imgs"
    
    os.makedirs(seg_masks_dir, exist_ok=True)
    os.makedirs(rectified_pcd_dir, exist_ok=True)
    os.makedirs(left_img_dir, exist_ok=True)
    os.makedirs(right_img_dir, exist_ok=True)
    
    success_count = 0
    total_count = 0
    
    bev_generator = BEVGenerator()
    
    pcd_files = []
    for root, _, files in os.walk(pcd_dir):
        for file in files:
            if file == "left-segmented-labelled.ply":
                pcd_files.append(os.path.join(root, file))
    
    total_count = len(pcd_files)
    
    pcd_files.sort()
    bev_generator = BEVGenerator()

    for idx, pcd_path in enumerate(tqdm(pcd_files, desc="Processing point clouds")):
        try:
            pcd_input = o3d.t.io.read_point_cloud(pcd_path)
            
            # saving left-imgs
            img_dir = os.path.dirname(pcd_path)
            left_src = os.path.join(img_dir, "left.jpg")
            right_src = os.path.join(img_dir, "right.jpg")

            left_dest = os.path.join(left_img_dir, f"left-img-{idx}.jpg")
            right_dest = os.path.join(right_img_dir, f"right-img-{idx}.jpg")

            shutil.copy(left_src, left_dest)
            shutil.copy(right_src, right_dest)

            # saving rectified pcd
            pcd_rectified = bev_generator.tilt_rectification(pcd_input)
            output_path = os.path.join(rectified_pcd_dir, f"rectified-pcd-{idx}.ply")
            o3d.t.io.write_point_cloud(output_path, pcd_rectified)

            # # generating GT-seg-mask
            crop_bb = {'x_min': -2.5, 'x_max': 2.5, 'z_min': 0, 'z_max': 5}
            seg_mask_mono , seg_mask_rgb = bev_generator.pcd_to_seg_mask(pcd_input, 
                                                                          nx = 256, nz = 256, 
                                                                          bb = crop_bb)
            seg_mask_mono = np.flip(seg_mask_mono, axis=0)
            
            # # saving GT-seg-mask
            output_path = os.path.join(seg_masks_dir, f"seg-mask-{idx}.png")
            plot_segmentation_classes(seg_mask_mono, output_path)
            
        except Exception as e:
            logger.error(f"Error processing {pcd_path}: {e}")

    # pcd_input = o3d.t.io.read_point_cloud(pcd_files[-1])
    # crop_bb = {'x_min': -10, 'x_max': 10, 'z_min': 0, 'z_max': 20}
    # seg_mask_mono , seg_mask_rgb = bev_generator.pcd_to_seg_mask(pcd_input, 
    #                                                                 nx = 256, nz = 256, 
    #                                                                 bb = crop_bb)
    
    # plot_segmentation_classes(seg_mask_mono)



    # ================================================
    # CASE 7: testing BEVGenerator
    # ================================================
    
    # pcd_input = o3d.t.io.read_point_cloud("debug/frames/frame-2686/left-segmented-labelled.ply")
    
    # bev_generator = BEVGenerator()
    # pcd_rectified: o3d.t.geometry.PointCloud = bev_generator.tilt_rectification(pcd_input)
    


    
    # ================================================
    # CASE 6: testing bev_generator.updated_camera_extrinsics()
    # ================================================
    
    # bev_generator = BEVGenerator()
    # ground_id = bev_generator.LABELS["NAVIGABLE_SPACE"]["id"]
    # camera_matrix = np.array([[1093.2768, 0, 964.989],
    #                            [0, 1093.2768, 569.276],
    #                            [0, 0, 1]], dtype=np.float32)
    
    # pcd_input = o3d.t.io.read_point_cloud("debug/left-segmented-labelled.ply")
    # pcd_rectified = bev_generator.tilt_rectification(pcd_input)
    
    # R = bev_generator.compute_tilt_matrix(pcd_input)
    # yaw_i, pitch_i, roll_i = bev_generator.rotation_matrix_to_ypr(R)
    
    # logger.info(f"================================================")
    # logger.info(f"yaw_i: {yaw_i}, pitch_i: {pitch_i}, roll_i: {roll_i}")
    # logger.info(f"================================================\n")
    
    # is_orthogonal = np.allclose(np.dot(R.T, R), np.eye(3), atol=1e-6)

    # logger.info(f"================================================")
    # logger.info(f"Is the rotation matrix orthogonal? {is_orthogonal}")
    # logger.info(f"================================================\n")

    # R_transpose = R.T
    # # logger.info(f"Transpose of R: \n{R_transpose}")
    # yaw_f, pitch_f, roll_f = bev_generator.rotation_matrix_to_ypr(R_transpose)
    
    # logger.info(f"================================================")
    # logger.info(f"R_transpose.shape: {R_transpose.shape}")
    # logger.info(f"yaw_f: {yaw_f}, pitch_f: {pitch_f}, roll_f: {roll_f}")
    # logger.info(f"================================================\n")

    # R_transpose_4x4 = np.eye(4)
    # R_transpose_4x4[:3, :3] = R_transpose

    # T_cam_world_i = np.eye(4)
    # T_cam_world_i = T_cam_world_i @ R_transpose_4x4

    # logger.info(f"================================================")
    # logger.info(f"T_cam_world_i: \n{T_cam_world_i}")
    # logger.info(f"================================================\n")
    
    # img_i = project_pcd_to_camera(pcd_input, camera_matrix, (1920, 1080), rvec=np.zeros((3, 1)), tvec=np.zeros((3, 1)))
    
    # rvec, _ = cv2.Rodrigues(R_transpose)
    # img_f = project_pcd_to_camera(pcd_rectified, camera_matrix, (1920, 1080), rvec=rvec, tvec=np.zeros((3, 1)))


    # ================================================
    # CASE 5: check image size
    # ================================================
    
    # img_path = "data/train-data/0/_left.jpg"  # Specify the path to your image
    # image = cv2.imread(img_path)
    # if image is not None:
    #     height, width, _ = image.shape
    #     logger.info(f"Image size: {width}x{height}")
    # else:
    #     logger.error("Failed to read the image.")


    # ================================================
    # CASE 4: testing pointcloud to camera projection
    # ================================================
   
    # images = read_images_binary("debug/dense-reconstruction/images.bin")

    # sparse_image_keys = images.keys()
    # sparse_image_keys = sorted(sparse_image_keys)

    # logger.info(f"================================================")
    # logger.info(f"sparse_image_keys: {sparse_image_keys}")
    # logger.info(f"================================================\n")

    # # for idx,key in enumerate(sparse_image_keys):
    # #     logger.info(f"================================================")
    # #     logger.info(f"idx: {idx} key: {key}")
    # #     left_img = images[key]
    # #     logger.info(f"left_img.name: {left_img.name}")
    # #     logger.info(f"================================================\n")

    # # exit()
    # num_sparse_images = len(sparse_image_keys)
    
    # logger.info(f"================================================")
    # logger.info(f"num_sparse_images: {num_sparse_images}")
    # logger.info(f"================================================\n")

    # left_img = images[sparse_image_keys[76]]

    # logger.info(f"================================================")
    # logger.info(f"left_img.name: {left_img.name}")
    # logger.info(f"================================================\n")

    # C_l = cam_extrinsics(left_img)
    # pcd_input = o3d.t.io.read_point_cloud("debug/dense-reconstruction/dense.ply")
    
    # # Extract point cloud positions and colors
    # points = pcd_input.point['positions'].numpy()
    # colors = pcd_input.point['colors'].numpy()  # Extract colors

    # logger.info(f"================================================")
    # logger.info(f"points.shape: {points.shape}")
    # logger.info(f"================================================\n")  

    # tvec_i = C_l[:3, 3]  # Extract translation vector from camera extrinsics
    # rvec_i = cv2.Rodrigues(C_l[:3, :3])[0]  # Extract rotation vector from rotation matrix

    # camera_matrix = np.array([[1090.536, 0, 954.99],
    #                            [0, 1090.536, 523.12],
    #                            [0, 0, 1]], dtype=np.float32)  # Camera intrinsic parameters

    # # Project points to camera
    # projected_points, _ = cv2.projectPoints(points, rvec=rvec_i, tvec=tvec_i, 
    #                                          cameraMatrix=camera_matrix, distCoeffs=None)
    
    # logger.info(f"================================================")
    # logger.info(f"projected_points.shape: {projected_points.shape}")
    # logger.info(f"================================================\n")

    # # projected_points = project_points_with_cv2(points, camera_matrix, (640, 720))
  
    # projected_points = projected_points.reshape(-1, 2)  # Reshape for OpenCV
    # img = np.zeros((1080, 1920, 3), dtype=np.uint8)  # Create a blank image

    
    # for idx, (point, color) in enumerate(zip(projected_points, colors)):
    #     x, y = int(point[0]), int(point[1])
    #     if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
    #         color_bgr = (int(color[2]), int(color[1]), int(color[0]))
    #         cv2.circle(img, (x, y), 3, color_bgr, -1)
    #         # 
    #         # if idx == 0:
    #             # logger.info(f"================================================")
    #             # logger.info(f"color: {color}")
    #             # logger.info(f"color_bgr: {color_bgr}")
    #             # logger.info(f"================================================\n")
    #         # 
    # cv2.imshow("Projected Points", img)
    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()


    # ================================================
    # CASE 2: testing CAMERA_PARAMS
    # ================================================

    # camera_params = get_zed_camera_params("debug/front_2024-06-05-09-14-54.svo")
    # logger.info(f"================================================")
    # logger.info(f"camera_params: {camera_params}")
    # logger.info(f"================================================\n")
    # exit()


    # # ================================================
    # # CASE 1: generate segmentation masks
    # # ================================================
    # vis = o3d.visualization.Visualizer()
    # bev_generator = BEVGenerator()
    
    # pcd_input = o3d.t.io.read_point_cloud("debug/left-segmented-labelled.ply")
    # pcd_rectified = bev_generator.tilt_rectification(pcd_input)
    
    # ground_id = bev_generator.LABELS["NAVIGABLE_SPACE"]["id"]
    
    # # ground plane normal => [original / rectified] pointcloud
    # # return [a, b, c, d]
    # n_i, _ = bev_generator.get_class_plane(pcd_input, ground_id)
    # n_f, _ = bev_generator.get_class_plane(pcd_rectified, ground_id)

    # # logger.info(f"================================================")
    # # logger.info(f"n_i.shape: {n_i.shape}")
    # # logger.info(f"n_f.shape: {n_f.shape}")
    # # logger.info(f"================================================\n")
    
    # # pitch, yaw, roll  => [original / rectified]
    # p_i, y_i, r_i = bev_generator.axis_angles(n_i)
    # p_f, y_f, r_f = bev_generator.axis_angles(n_f)

    # logger.info(f"================================================")
    # logger.info(f"[BEFORE RECTIFICATION] - Yaw: {y_i:.2f}, Pitch: {p_i:.2f}, Roll: {r_i:.2f}")
    # logger.info(f"[AFTER RECTIFICATION] - Yaw: {y_f:.2f}, Pitch: {p_f:.2f}, Roll: {r_f:.2f}")
    # logger.info(f"================================================\n")

    # # generate BEV
    # bev_pcd = bev_generator.generate_BEV(pcd_input)
    
    # logger.info(f"================================================")
    # logger.info(f"Number of points in bev_pcd: {len(bev_pcd.point['positions'].numpy())}")
    # logger.info(f"================================================\n")
    
    # # cropping params
    # crop_bb = {'x_min': -5, 'x_max': 5, 'z_min': 0, 'z_max': 10}
    
    # bev_pcd_cropped = crop_pcd(bev_pcd, crop_bb)

    # x_values = bev_pcd_cropped.point['positions'][:, 0].numpy()
    # y_values = bev_pcd_cropped.point['positions'][:, 1].numpy()
    # z_values = bev_pcd_cropped.point['positions'][:, 2].numpy()
    
    # logger.info(f"================================================")
    # logger.info(f"Range of x values: {x_values.min()} to {x_values.max()}")
    # logger.info(f"Range of y values: {y_values.min()} to {y_values.max()}")
    # logger.info(f"Range of z values: {z_values.min()} to {z_values.max()}")
    # logger.info(f"================================================\n")
    
    # seg_mask_mono = bev_generator.bev_to_seg_mask_mono(bev_pcd_cropped, 
    #                                                             nx = 400, nz = 400, 
    #                                                             bb = crop_bb)
    # seg_mask_rgb = mono_to_rgb_mask(seg_mask_mono)
    # cv2.imshow("seg_mask_rgb", seg_mask_rgb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # logger.info(f"================================================")
    # logger.info(f"seg_mask_rgb.shape: {seg_mask_rgb.shape}")
    # logger.info(f"================================================\n")

    # output_path = "debug/seg-mask-rgb.png"
    
    # cv2.imwrite(output_path, seg_mask_rgb)

    # logger.info(f"================================================")
    # logger.info(f"Segmentation mask saved to {output_path}")
    # logger.info(f"================================================\n")

    # ================================================
    # visualization
    # ================================================
    # vis.create_window()
        
    # # Co-ordinate frame for vis window      
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])
    # vis.add_geometry(coordinate_frame)
    
    # # Adding point clouds to visualizer
    # # vis.add_geometry(combined_pcd.to_legacy())
    # vis.add_geometry(bev_pcd_cropped.to_legacy())
    
    # view_ctr = vis.get_view_control()
    # view_ctr.set_front(np.array([0, -1, 0]))
    # view_ctr.set_up(np.array([0, 0, 1]))
    # # view_ctr.set_zoom(0.9)
    # view_ctr.set_zoom(4)
    
    # vis.run()
    # vis.destroy_window()

    # ================================================
    # CASE 0: testing bev_generator.pcd_to_seg_mask_mono()
    # ================================================

    # src_folder = "train-data"
    # dst_folder = "debug/output-seg-masks"
    # bev_generator = BEVGenerator()

    # crop_bb = {'x_min': -5, 'x_max': 5, 'z_min': 0, 'z_max': 10}
    # nx = 400
    # nz = 400

    # if not os.path.exists(dst_folder):
    #     os.makedirs(dst_folder)

    # left_segmented_labelled_files = []

    # total_files = sum(len(files) for _, _, files in os.walk(src_folder) if 'left-segmented-labelled.ply' in files)
    # with tqdm(total=total_files, desc="Processing files", ncols=100) as pbar:
    #     for root, dirs, files in os.walk(src_folder):
    #         for file in files:
    #             if file == 'left-segmented-labelled.ply':
    #                 file_path = os.path.join(root, file)
    #                 left_segmented_labelled_files.append(file_path)

    #                 try:
    #                     pcd_input = o3d.t.io.read_point_cloud(file_path)
    #                     seg_mask_mono, seg_mask_rgb = bev_generator.pcd_to_seg_mask(pcd_input, 
    #                                                                                  nx=nx, nz=nz, 
    #                                                                                  bb=crop_bb)

    #                     output_rgb_path = os.path.join(dst_folder, f"seg-mask-rgb-{os.path.basename(root)}.png")
    #                     cv2.imwrite(output_rgb_path, seg_mask_rgb)
    #                 except Exception as e:
    #                     logger.error(f"Error processing {file_path}: {e}")
    
    #                 pbar.update(1)
