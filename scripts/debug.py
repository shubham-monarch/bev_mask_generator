#! /usr/bin/env python3

import open3d as o3d
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np
import yaml
import logging
import shutil
import traceback

from scripts.bev_mask_generator_dairy import BEVGenerator
from scripts.logger import get_logger
from scripts.occ_mask_generator import OccMap
from scripts.data_generator_s3 import DataGeneratorS3, LeafFolder

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

    # ==========================
    # CASE 10: occ-mask generation
    # ==========================

    pcd_dir = f"debug/frames-3"
    
    left_img_dir = f"debug/3/left-imgs"
    seg_masks_dir = f"debug/3/seg-masks"
    segmented_pcd_dir = f"debug/3/segmented-pcd"
    
    os.makedirs(seg_masks_dir, exist_ok=True)
    os.makedirs(segmented_pcd_dir, exist_ok=True)
    os.makedirs(left_img_dir, exist_ok=True)
    
    # assert not (os.path.exists(seg_masks_dir) and os.listdir(seg_masks_dir)), f"{seg_masks_dir} must be empty if it exists."
    # assert not (os.path.exists(segmented_pcd_dir) and os.listdir(segmented_pcd_dir)), f"{segmented_pcd_dir} must be empty if it exists."
    # assert not (os.path.exists(left_img_dir) and os.listdir(left_img_dir)), f"{left_img_dir} must be empty if it exists."

    # bev_generator = BEVGenerator()
    
    # pcd_files = []
    # for root, _, files in os.walk(pcd_dir):
    #     for file in files:
    #         if file == "left-segmented-labelled.ply":
    #             pcd_files.append(os.path.join(root, file))
    
    # total_count = len(pcd_files)
    # # random.shuffle(pcd_files)
    # pcd_files.sort()

    # occ_map_generator = OcclusionMap()

    void_counter = []

    # src_URIs = ["s3://occupancy-dataset/occ-dataset/vineyards/gallo/", 
    #             "s3://occupancy-dataset/occ-dataset/vineyards/RJM/"]
    
    src_URIs = ["s3://occupancy-dataset/occ-dataset/dairy/"]

    leaf_folders = DataGeneratorS3.get_leaf_folders(src_URIs)

    logger.info(f"===========================")
    logger.info(f"len(leaf_folders): {len(leaf_folders)}")    
    logger.info(f"===========================\n")

    counter = 0
    random.seed(2)  # Set a seed for reproducibility
    
    for i in range(10):
        # try:      

        logger.info("────" * 10)
        logger.info(f"i: {i}")
        logger.info(f"────" * 10)
        # continue
        
        # leaf_folder_uri = random.choice(leaf_folders)
        
        # logger.info(f"===========================")
        # logger.info(f"{i} ==> [{leaf_folder_uri}]")
        # logger.info(f"===========================\n")

        # # save left-img
        # left_img_uri = os.path.join(leaf_folder_uri, "left.jpg")
        # left_img_TMP = LeafFolder.download_file(left_img_uri, log_level=logging.WARNING)
        # left_img_DEST = os.path.join(left_img_dir, f"left-img-{i}.jpg")
        # shutil.copy(left_img_TMP, left_img_DEST)
        
        # save left-segmented-labelled.ply
        # left_segmented_labelled_uri = os.path.join(leaf_folder_uri, "left-segmented-labelled.ply")
        # sfm_pcd_TMP = LeafFolder.download_file(left_segmented_labelled_uri, log_level=logging.WARNING)
        # ssfm_pcd_DEST = os.path.join(segmented_pcd_dir, f"sfm-pcd-{i}.ply")
        # shutil.copy(sfm_pcd_TMP, sfm_pcd_DEST)

        sfm_pcd_DEST_PATH = f"debug/3/segmented-pcd/sfm-pcd-{i}.ply"    

        logger.info("────" * 10)
        logger.info(f"sfm_pcd_DEST_PATH: {sfm_pcd_DEST_PATH}")
        logger.info("────" * 10 + "\n")
        
        # # bev-generator
        sfm_pcd = o3d.t.io.read_point_cloud(sfm_pcd_DEST_PATH)
        
        bev_generator = BEVGenerator(logging_level=logging.INFO, yaml_path="config/dairy.yaml")
        
        # for label_id in range(14):
        #     pcd_class = bev_generator.get_class_pointcloud(sfm_pcd, label_id)
        #     # o3d.t.io.write_point_cloud(os.path.join(segmented_pcd_dir, f"sfm-pcd-{i}-{label_id}.ply"), pcd_class)
        
        
        # bev_generator = BEVGenerator(logging_level=logging.INFO, yaml_path="config/dairy.yaml")
        # # bev_generator.generate_pcd_BEV_2D(sfm_pcd)
        bev_generator.count_unique_labels(sfm_pcd)
        
        # seg_mask_mono, seg_mask_rgb = bev_generator.pcd_to_seg_mask(sfm_pcd, 
        #                                                             nx = 256, 
        #                                                             nz = 256, 
        #                                                             bb = {'x_min': -2.5, 'x_max': 2.5, 'z_min': 0.02, 'z_max': 5.02})

        # seg_mask_mono = np.flip(seg_mask_mono, axis=0)
        # seg_mask_rgb = np.flip(seg_mask_rgb, axis=0)


        
        # cv2.imwrite(os.path.join(seg_masks_dir, f"seg-mask-mono-{i}.png"), seg_mask_mono)
        # cv2.imwrite(os.path.join(seg_masks_dir, f"seg-mask-rgb-{i}.png"), seg_mask_rgb)
        
  
    

    # # ==========================
    # # CASE 9: bev_mask_generation -- DAIRY
    # # ==========================

    # pcd_dir = f"debug/frames-3"
    
    # left_img_dir = f"debug/3/left-imgs"
    # seg_masks_dir = f"debug/3/seg-masks"
    # segmented_pcd_dir = f"debug/3/segmented-pcd"
    
    # os.makedirs(seg_masks_dir, exist_ok=True)
    # os.makedirs(segmented_pcd_dir, exist_ok=True)
    # os.makedirs(left_img_dir, exist_ok=True)
    
    # # assert not (os.path.exists(seg_masks_dir) and os.listdir(seg_masks_dir)), f"{seg_masks_dir} must be empty if it exists."
    # # assert not (os.path.exists(segmented_pcd_dir) and os.listdir(segmented_pcd_dir)), f"{segmented_pcd_dir} must be empty if it exists."
    # # assert not (os.path.exists(left_img_dir) and os.listdir(left_img_dir)), f"{left_img_dir} must be empty if it exists."

    # # bev_generator = BEVGenerator()
    
    # # pcd_files = []
    # # for root, _, files in os.walk(pcd_dir):
    # #     for file in files:
    # #         if file == "left-segmented-labelled.ply":
    # #             pcd_files.append(os.path.join(root, file))
    
    # # total_count = len(pcd_files)
    # # # random.shuffle(pcd_files)
    # # pcd_files.sort()

    # # occ_map_generator = OcclusionMap()

    # void_counter = []

    # # src_URIs = ["s3://occupancy-dataset/occ-dataset/vineyards/gallo/", 
    # #             "s3://occupancy-dataset/occ-dataset/vineyards/RJM/"]
    
    # src_URIs = ["s3://occupancy-dataset/occ-dataset/dairy/"]

    # leaf_folders = DataGeneratorS3.get_leaf_folders(src_URIs)

    # logger.info(f"===========================")
    # logger.info(f"len(leaf_folders): {len(leaf_folders)}")    
    # logger.info(f"===========================\n")

    # counter = 0
    # random.seed(2)  # Set a seed for reproducibility
    
    # for i in range(10):
    #     # try:      

    #     logger.info("────" * 10)
    #     logger.info(f"i: {i}")
    #     logger.info(f"────" * 10)
    #     # continue
        
    #     # leaf_folder_uri = random.choice(leaf_folders)
        
    #     # logger.info(f"===========================")
    #     # logger.info(f"{i} ==> [{leaf_folder_uri}]")
    #     # logger.info(f"===========================\n")

    #     # # save left-img
    #     # left_img_uri = os.path.join(leaf_folder_uri, "left.jpg")
    #     # left_img_TMP = LeafFolder.download_file(left_img_uri, log_level=logging.WARNING)
    #     # left_img_DEST = os.path.join(left_img_dir, f"left-img-{i}.jpg")
    #     # shutil.copy(left_img_TMP, left_img_DEST)
        
    #     # save left-segmented-labelled.ply
    #     # left_segmented_labelled_uri = os.path.join(leaf_folder_uri, "left-segmented-labelled.ply")
    #     # sfm_pcd_TMP = LeafFolder.download_file(left_segmented_labelled_uri, log_level=logging.WARNING)
    #     # ssfm_pcd_DEST = os.path.join(segmented_pcd_dir, f"sfm-pcd-{i}.ply")
    #     # shutil.copy(sfm_pcd_TMP, sfm_pcd_DEST)

    #     sfm_pcd_DEST_PATH = f"debug/3/segmented-pcd/sfm-pcd-{i}.ply"    

    #     logger.info("────" * 10)
    #     logger.info(f"sfm_pcd_DEST_PATH: {sfm_pcd_DEST_PATH}")
    #     logger.info("────" * 10 + "\n")
        
    #     # # bev-generator
    #     sfm_pcd = o3d.t.io.read_point_cloud(sfm_pcd_DEST_PATH)
        
    #     bev_generator = BEVGenerator(logging_level=logging.INFO, yaml_path="config/dairy.yaml")
        
    #     # for label_id in range(14):
    #     #     pcd_class = bev_generator.get_class_pointcloud(sfm_pcd, label_id)
    #     #     # o3d.t.io.write_point_cloud(os.path.join(segmented_pcd_dir, f"sfm-pcd-{i}-{label_id}.ply"), pcd_class)
        
        
    #     # bev_generator = BEVGenerator(logging_level=logging.INFO, yaml_path="config/dairy.yaml")
    #     # # bev_generator.generate_pcd_BEV_2D(sfm_pcd)
    #     bev_generator.count_unique_labels(sfm_pcd)
        
    #     # seg_mask_mono, seg_mask_rgb = bev_generator.pcd_to_seg_mask(sfm_pcd, 
    #     #                                                             nx = 256, 
    #     #                                                             nz = 256, 
    #     #                                                             bb = {'x_min': -2.5, 'x_max': 2.5, 'z_min': 0.02, 'z_max': 5.02})

    #     # seg_mask_mono = np.flip(seg_mask_mono, axis=0)
    #     # seg_mask_rgb = np.flip(seg_mask_rgb, axis=0)


        
    #     # cv2.imwrite(os.path.join(seg_masks_dir, f"seg-mask-mono-{i}.png"), seg_mask_mono)
    #     # cv2.imwrite(os.path.join(seg_masks_dir, f"seg-mask-rgb-{i}.png"), seg_mask_rgb)
        
  
     

    # ================================================
    # CASE 9: occ-mask generation
    # ================================================

    # pcd_dir = f"debug/frames-3"
    
    # left_img_dir = f"debug/3/left-imgs"
    # right_img_dir = f"debug/3/right-imgs"
    # seg_masks_dir = f"debug/3/seg-masks"
    # pcd_RECTIFIED_dir = f"debug/3/rectified-pcd"
    # pcd_OCC_dir = f"debug/3/occ-pcd"
    # pcd_BEV_3D_dir = f"debug/3/bev-3d-pcd"

    # os.makedirs(seg_masks_dir, exist_ok=True)
    # os.makedirs(pcd_RECTIFIED_dir, exist_ok=True)
    # os.makedirs(left_img_dir, exist_ok=True)
    # os.makedirs(right_img_dir, exist_ok=True)
    # os.makedirs(pcd_OCC_dir, exist_ok=True)
    # os.makedirs(pcd_BEV_3D_dir, exist_ok=True)

    # # assert not (os.path.exists(seg_masks_dir) and os.listdir(seg_masks_dir)), f"{seg_masks_dir} must be empty if it exists."
    # # assert not (os.path.exists(pcd_RECTIFIED_dir) and os.listdir(pcd_RECTIFIED_dir)), f"{pcd_RECTIFIED_dir} must be empty if it exists."
    # # assert not (os.path.exists(left_img_dir) and os.listdir(left_img_dir)), f"{left_img_dir} must be empty if it exists."
    # # assert not (os.path.exists(right_img_dir) and os.listdir(right_img_dir)), f"{right_img_dir} must be empty if it exists."
    # # assert not (os.path.exists(pcd_OCC_dir) and os.listdir(pcd_OCC_dir)), f"{pcd_OCC_dir} must be empty if it exists."
    # # assert not (os.path.exists(pcd_BEV_3D_dir) and os.listdir(pcd_BEV_3D_dir)), f"{pcd_BEV_3D_dir} must be empty if it exists."
    
    # success_count = 0
    # total_count = 0
    
    # bev_generator = BEVGenerator()
    
    # pcd_files = []
    # for root, _, files in os.walk(pcd_dir):
    #     for file in files:
    #         if file == "left-segmented-labelled.ply":
    #             pcd_files.append(os.path.join(root, file))
    
    # total_count = len(pcd_files)
    # # random.shuffle(pcd_files)
    # pcd_files.sort()

    # # occ_map_generator = OcclusionMap()

    # void_counter = []

    # src_URIs = ["s3://occupancy-dataset/occ-dataset/vineyards/gallo/"]
    # leaf_folders = DataGeneratorS3.get_leaf_folders(src_URIs)

    # logger.info(f"===========================")
    # logger.info(f"len(leaf_folders): {len(leaf_folders)}")    
    # logger.info(f"===========================\n")

    # for idx, pcd_path in enumerate(tqdm(pcd_files, desc="Processing point clouds")):
    #     try:

    #         # logger.warning(f"===========================")
    #         # logger.warning(f"{idx} => {pcd_path}")
    #         # logger.warning(f"===========================\n")

    #         leaf_URI = random.choice(leaf_folders)
            

    #         # if idx > 0:
    #         #     break
    #         # pcd_path = random.choice(pcd_files)
    #         # pcd_path = "debug/frames-3/frame-1240/left-segmented-labelled.ply"
    #         # pcd_path = "debug/frames-4/frame-608/left-segmented-labelled.ply"
            
    #         # case -> heavy occlusion
    #         # pcd_path = "debug/frames-4/frame-568/left-segmented-labelled.ply"
            
            
    #         # # read pcd
    #         # PCD_INPUT = o3d.t.io.read_point_cloud(pcd_path)
    #         # PCD = PCD_INPUT.clone()
            
    #         # # camera-matrix 
    #         # CAMERA_INTRINSIC_MATRIX = np.array([[1090.536, 0, 954.99],
    #         #                            [0, 1090.536, 523.12],
    #         #                            [0, 0, 1]], dtype=np.float32)
    #         # # crop-bb 
    #         # CROP_BB = {'x_min': -2.5, 'x_max': 2.5, 'z_min': 0.02, 'z_max': 5.02}
    #         # # CROP_BB = {'x_min': -2.5, 'x_max': 2.5, 'z_min': 1.5, 'z_max': 6.5}

    #         # saving left / right imgs
    #         img_dir = os.path.dirname(pcd_path)
    #         left_src = os.path.join(img_dir, "left.jpg")
    #         right_src = os.path.join(img_dir, "right.jpg")

    #         left_img = cv2.imread(left_src)
    #         right_img = cv2.imread(right_src)

    #         left_img = cv2.resize(left_img, (480, 640))
    #         right_img = cv2.resize(right_img, (480, 640))

    #         logger.info(f"===========================")
    #         logger.info(f"left_img.shape: {left_img.shape}")
    #         logger.info(f"right_img.shape: {right_img.shape}")
    #         logger.info(f"===========================\n")

    #         is_rectified, avg_vertical_disparity = OccMap.is_rectified(left_img, right_img)

    #         break
    #         # left_dest = os.path.join(left_img_dir, f"left-img-{idx}.jpg")
    #         # right_dest = os.path.join(right_img_dir, f"right-img-{idx}.jpg")

    #         # shutil.copy(left_src, left_dest)
    #         # shutil.copy(right_src, right_dest)

    #         # # initialize bev_generator
    #         # bev_generator = BEVGenerator(logging_level=logging.ERROR)
    #         # seg_mask_mono , seg_mask_rgb = bev_generator.pcd_to_seg_mask(PCD, 
    #         #                                                               nx = 256, nz = 256, 
    #         #                                                               bb = CROP_BB,
    #         #                                                               yaml_path="config/Mavis.yaml")
            
    #         # # seg_mask_mono = np.flip(seg_mask_mono, axis=0)
    #         # # seg_mask_rgb = np.flip(seg_mask_rgb, axis=0)

    #         # # saving seg-mask-rgb
    #         # path_seg_mask_rgb = os.path.join(seg_masks_dir, f"seg-mask-rgb-{idx}.png")
    #         # cv2.imwrite(path_seg_mask_rgb, seg_mask_rgb)

        
    #         # # saving 3D BEV-pcd
    #         # pcd_BEV_3D = bev_generator.get_pcd_BEV_3D()
    #         # path_pcd_BEV_3D = os.path.join(pcd_BEV_3D_dir, f"pcd-BEV-3D-{idx}.ply")
    #         # o3d.t.io.write_point_cloud(path_pcd_BEV_3D, pcd_BEV_3D)


    #         # # occ-map generation
    #         # BEV_3D_PCD = bev_generator.get_pcd_BEV_3D()
            
    #         # # 4x4 camera extrinsic matrix [R | t]
    #         # CAMERA_EXTRINIC_MATRIX = bev_generator.get_updated_camera_extrinsics() 
    #         # # CAMERA_EXTRINIC_MATRIX = np.eye(4)
    #         # pcd_OCC = OccMap.get_occ_pcd(pcd = BEV_3D_PCD, 
    #         #                               K = CAMERA_INTRINSIC_MATRIX,
    #         #                               P = CAMERA_EXTRINIC_MATRIX[:3, :],
    #         #                               bb = CROP_BB,
    #         #                               to_crop = False)
            
    #         # # saving occ-pcd
    #         # path_pcd_OCC = os.path.join(pcd_OCC_dir, f"pcd-OCC-{idx}.ply")
    #         # o3d.t.io.write_point_cloud(path_pcd_OCC, pcd_OCC)

            
    #         # # plotting GT-seg-mask
    #         # # plot_segmentation_classes(seg_mask_mono, output_path)
    #         # # plot_segmentation_classes(seg_mask_mono)
             
    #     except Exception as e:
    #         logger.error(f"================================================")
    #         logger.error(f"Error processing {pcd_path} with error {e}")
    #         logger.error(f"================================================\n")
    
  


    # # # ================================================
    # # # CASE 8: occ-map generation [aws version]
    # # # ================================================

    # pcd_dir = f"debug/frames-4"
    
    # seg_masks_dir = f"debug/4/seg-masks"
    # rectified_pcd_dir = f"debug/4/rectified-pcd"
    # left_img_dir = f"debug/4/left-imgs"
    # right_img_dir = f"debug/4/right-imgs"
    # occ_pcd_dir = f"debug/4/occ-pcd"

    # os.makedirs(seg_masks_dir, exist_ok=True)
    # os.makedirs(rectified_pcd_dir, exist_ok=True)
    # os.makedirs(left_img_dir, exist_ok=True)
    # os.makedirs(right_img_dir, exist_ok=True)
    # os.makedirs(occ_pcd_dir, exist_ok=True)
    
    # # assert not (os.path.exists(seg_masks_dir) and os.listdir(seg_masks_dir)), f"{seg_masks_dir} must be empty if it exists."
    # # assert not (os.path.exists(rectified_pcd_dir) and os.listdir(rectified_pcd_dir)), f"{rectified_pcd_dir} must be empty if it exists."
    # # assert not (os.path.exists(left_img_dir) and os.listdir(left_img_dir)), f"{left_img_dir} must be empty if it exists."
    # # assert not (os.path.exists(right_img_dir) and os.listdir(right_img_dir)), f"{right_img_dir} must be empty if it exists."
    # # assert not (os.path.exists(occ_pcd_dir) and os.listdir(occ_pcd_dir)), f"{occ_pcd_dir} must be empty if it exists."
    
    # success_count = 0
    # total_count = 0
    
    # bev_generator = BEVGenerator()
    
    # pcd_files = []
    # for root, _, files in os.walk(pcd_dir):
    #     for file in files:
    #         if file == "left-segmented-labelled.ply":
    #             pcd_files.append(os.path.join(root, file))
    
    # total_count = len(pcd_files)
    # # random.shuffle(pcd_files)
    # pcd_files.sort()

    # bev_generator = BEVGenerator(logging_level=logging.WARNING)
    # occ_map_generator = OcclusionMap()

    # OCC_FOLDER_URI = ["s3://occupancy-dataset/occ-dataset/vineyards/gallo/"]
            
    # data_generator_s3 = DataGeneratorS3(
    #     src_URIs=OCC_FOLDER_URI, 
    #     dest_folder=None,
    #     index_json="dummy",
    #     color_map="dummy",
    #     crop_bb={"x": 0, "y": 0, "w": 1, "h": 1, "d": 1},
    #     nx=1,
    #     nz=1
    # )

    # for idx, pcd_path in enumerate(tqdm(pcd_files, desc="Processing point clouds")):
    #     try:
    #         # if idx > 0:
    #         #     break
    #         # pcd_path = random.choice(pcd_files)
            
           
        
    #         leaf_URIs = data_generator_s3.get_leaf_folders()
    #         leaf_URI = random.choice(leaf_URIs)

    #         logger.info(f"================================================")
    #         logger.info(f"leaf_URI: {leaf_URI}")
    #         logger.info(f"================================================\n")


    #         leaf_folder = LeafFolder(src_URI=leaf_URI, dest_URI="dummy_dest", index_json="dummy_index.json", crop_bb={"x": 0, "y": 0, "z": 0, "w": 1, "h": 1, "d": 1}, color_map="dummy_color_map", nx=1, nz=1)
            
    #         # download left-segmented-labelled.ply
    #         left_segmented_labelled_pcd_URI = os.path.join(leaf_URI, "left-segmented-labelled.ply")
    #         pcd_path = leaf_folder.download_file(left_segmented_labelled_pcd_URI)

    #         # read pcd
    #         pcd_input = o3d.t.io.read_point_cloud(pcd_path)
            
            
    #         # download left / right imgs
    #         left_img_URI = os.path.join(leaf_URI, "left.jpg")
    #         right_img_URI = os.path.join(leaf_URI, "right.jpg")
            
    #         left_img_path = leaf_folder.download_file(left_img_URI)
    #         right_img_path = leaf_folder.download_file(right_img_URI)
            
    #         # saving left / right imgs

    #         left_dest = os.path.join(left_img_dir, f"left-img-{idx}.jpg")
    #         right_dest = os.path.join(right_img_dir, f"right-img-{idx}.jpg")

    #         shutil.copy(left_img_path, left_dest)
    #         shutil.copy(right_img_path, right_dest)

    #         # saving rectified pcd
    #         pcd_rectified = bev_generator.get_tilt_rectified_pcd(pcd_input)
    #         path_pcd_rectified = os.path.join(rectified_pcd_dir, f"rectified-pcd-{idx}.ply")
    #         o3d.t.io.write_point_cloud(path_pcd_rectified, pcd_rectified)

    #         # camera-matrix 
    #         camera_matrix = np.array([[1090.536, 0, 954.99],
    #                                    [0, 1090.536, 523.12],
    #                                    [0, 0, 1]], dtype=np.float32)
    #         # crop-bb 
    #         CROP_BB = {'x_min': -2.49, 'x_max': 2.49, 'z_min': 0.02, 'z_max': 5}

    #         # occ-map generation
    #         bev_pcd = bev_generator.generate_BEV(pcd_rectified)
    #         pcd_downsampled = bev_generator.get_downsampled_pcd()
    #         camera_projection_matrix = bev_generator.get_updated_camera_extrinsics(pcd_input)
            
    #         occ_pcd = OcclusionMap.get_occ_mask(pcd = pcd_downsampled, 
    #                                                  camera_matrix = camera_matrix, 
    #                                                  bb = CROP_BB,
    #                                                  camera_projection_matrix = camera_projection_matrix)
    #         path_occ_pcd = os.path.join(occ_pcd_dir, f"occ-pcd-{idx}.ply")
    #         o3d.t.io.write_point_cloud(path_occ_pcd, occ_pcd)

    #         # break
    #         # generating GT-seg-mask
    #         # crop_bb = {'x_min': -2.5s, 'x_max': 2.5, 'z_min': 0, 'z_max': 5}
    #         seg_mask_mono , seg_mask_rgb = bev_generator.pcd_to_seg_mask(pcd_input, 
    #                                                                       nx = 256, nz = 256, 
    #                                                                       bb = CROP_BB,
    #                                                                       yaml_path="config/Mavis.yaml")
            
    #         seg_mask_mono = np.flip(seg_mask_mono, axis=0)
    #         seg_mask_rgb = np.flip(seg_mask_rgb, axis=0)

    #         # saving seg-mask-rgb
    #         path_seg_mask_rgb = os.path.join(seg_masks_dir, f"seg-mask-rgb-{idx}.png")
    #         cv2.imwrite(path_seg_mask_rgb, seg_mask_rgb)

            
    #         # plotting GT-seg-mask
    #         # plot_segmentation_classes(seg_mask_mono, output_path)
    #         # plot_segmentation_classes(seg_mask_mono)
             
    #     except Exception as e:
    #         logger.error(f"================================================")
    #         logger.error(f"Error processing {pcd_path} with error {e}")
    #         logger.error("Traceback:")
    #         logger.error(traceback.format_exc())
    #         logger.error(f"================================================\n")
    
    



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
