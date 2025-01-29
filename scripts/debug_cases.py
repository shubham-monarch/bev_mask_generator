"""Debug test cases implementation"""

import logging
import os
import random
import shutil
import traceback
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

from scripts.logger import get_logger
from scripts.dairy_mask_generator import BEVGenerator
from scripts.data_generator_s3 import DataGeneratorS3, LeafFolder
from scripts.occ_mask_generator import OccMap
# from scripts.helpers import get_zed_camera_params, cam_extrinsics, read_images_binary


logger = get_logger("debug_cases")


def test_dairy_masks():
    """Case 9: Test dairy mask generation"""
    
    pcd_dir = Path("debug/frames-11")
    output_dir = Path("debug/11")
    output_dirs = {     
        "left_img": output_dir / "left-imgs",
        "right_img": output_dir / "right-imgs",
        
        "sfm-pcd": output_dir / "sfm-pcd",
        "bev-pcd": output_dir / "bev-pcd",
        
        "seg-masks-rgb": output_dir / "seg-masks-rgb",
        "seg-masks-mono": output_dir / "seg-masks-mono",

        "labelled-pcd": output_dir / "labelled-pcd"
        
    }

    # output_dirs should be empty
    for dir_path in output_dirs.values():
        if dir_path.exists():
            assert not any(dir_path.iterdir()), f"Output directory {dir_path} is not empty."


    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # find all pcd files
    pcd_files = []
    for root, _, files in os.walk(pcd_dir):
        for file in files:
            if file == "left-segmented-labelled.ply":
                pcd_files.append(Path(root) / file)
    
    pcd_files.sort()
    
    # camera parameters
    camera_matrix = np.array([[1090.536, 0, 954.99],
                            [0, 1090.536, 523.12],
                            [0, 0, 1]], dtype=np.float32)
    

    # index_list = [0, 1, 2, 6, 7, 12, 13]
    # index_list = [6,7,12,13]
    # index_list = [5,6,7]
    index_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    
    # process each pcd file
    for idx in tqdm(index_list, desc="Processing point clouds"):
        try:
            pcd_path = pcd_files[idx]
            
            logger.warning("───────────────────────────────")
            logger.warning(f"IDX: {idx}")
            logger.warning("───────────────────────────────")
            
           
            # save segmented-sfm point cloud
            sfm_pcd = o3d.t.io.read_point_cloud(str(pcd_path))
            o3d.t.io.write_point_cloud(str(output_dirs["sfm-pcd"] / f"sfm-pcd-{idx}.ply"), sfm_pcd)
                        
            # Generate and save individual class point clouds
             # read labels from yaml file
            with open(f"config/dairy.yaml", 'r') as file:
                dairy_config = yaml.safe_load(file)
            
            # get labels from yaml config
            labels = sorted(dairy_config['labels'].keys())
            labels.append(0)
            
            for label_id in labels:
                mask = sfm_pcd.point["label"] == label_id
                pcd_class = sfm_pcd.select_by_index(mask.nonzero()[0])
                output_path = output_dirs["labelled-pcd"] / f"sfm-pcd-{idx}-{label_id}.ply"
                o3d.t.io.write_point_cloud(str(output_path), pcd_class)
            

        except Exception as e:
            logger.error(f"Error processing {pcd_files[idx]}: {str(e)}")
            logger.error(traceback.format_exc())

def test_stereo_pcd_occ():
    """Case 12: Test stereo point cloud occlusion map generation"""
    
    pcd_dir = Path("debug/frames-4")
    output_dir = Path("debug/4")
    output_dirs = { 
        "left_img": output_dir / "left-imgs",
        "right_img": output_dir / "right-imgs",
        
        "stereo_pcd": output_dir / "stereo-pcd",
        "sfm_pcd": output_dir / "sfm-pcd",
        
        "stereo_occ_pcd": output_dir / "stereo-occ-pcd",
        "sfm_occ_pcd": output_dir / "sfm-occ-pcd",
        # "combined_pcd": output_dir / "combined-pcd",
        
        "stereo_img": output_dir / "stereo-img",
        "sfm_img": output_dir / "sfm-img",
        
        "seg_masks_rgb": output_dir / "seg-masks-rgb",
        "seg_masks_mono": output_dir / "seg-masks-mono",
        
        "rectified_stereo_occ_pcd": output_dir / "rectified-stereo-occ-pcd",
        
        "rectified_stereo_pcd": output_dir / "rectified-stereo-pcd",
        "rectified_sfm_pcd": output_dir / "rectified-sfm-pcd", 

        "combined_rectified_pcd": output_dir / "combined-rectified-pcd",
        "combined_pcd": output_dir / "combined-pcd"
    }

    # output_dirs should be empty
    for dir_path in output_dirs.values():
        if dir_path.exists():
            assert not any(dir_path.iterdir()), f"Output directory {dir_path} is not empty."


    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # find all pcd files
    pcd_files = []
    for root, _, files in os.walk(pcd_dir):
        for file in files:
            if file == "left-segmented-labelled.ply":
                pcd_files.append(Path(root) / file)
    
    pcd_files.sort()
    
    # camera parameters
    camera_matrix = np.array([[1090.536, 0, 954.99],
                            [0, 1090.536, 523.12],
                            [0, 0, 1]], dtype=np.float32)
    

    # index_list = [0, 1, 2, 6, 7, 12, 13]
    # index_list = [6,7,12,13]
    index_list = [5,6,7]
    # index_list = [0]
    
    # process each pcd file
    for idx in tqdm(index_list, desc="Processing point clouds"):
        try:
            pcd_path = pcd_files[idx]
            
            logger.warning("───────────────────────────────")
            logger.warning(f"IDX: {idx}")
            logger.warning("───────────────────────────────")
            
            # read and process images
            img_dir = pcd_path.parent
            left_src = img_dir / "left.jpg"
            right_src = img_dir / "right.jpg"
            
            left_img = cv2.imread(str(left_src))
            right_img = cv2.imread(str(right_src))
            
            # save processed images
            left_dest = output_dirs["left_img"] / f"left-img-{idx}.jpg"
            right_dest = output_dirs["right_img"] / f"right-img-{idx}.jpg"
            
            cv2.imwrite(str(left_dest), left_img)
            cv2.imwrite(str(right_dest), right_img)
            
            # save segmented-sfm point cloud
            sfm_pcd = o3d.t.io.read_point_cloud(str(pcd_path))
            o3d.t.io.write_point_cloud(str(output_dirs["sfm_pcd"] / f"sfm-pcd-{idx}.ply"), sfm_pcd)
            
            
            # generate seg-bev masks
            bev_generator = BEVGenerator(logging_level=logging.WARNING, yaml_path="config/dairy.yaml")
            seg_mask_mono, seg_mask_rgb = bev_generator.pcd_to_seg_mask(
                sfm_pcd, nx=256, nz=256,
                bb={'x_min': -2.5, 'x_max': 2.5, 'z_min': 2.0, 'z_max': 7.0}
            )

            cv2.imwrite(str(output_dirs["seg_masks_rgb"] / f"seg-mask-rgb-{idx}.png"), seg_mask_rgb)
            cv2.imwrite(str(output_dirs["seg_masks_mono"] / f"seg-mask-mono-{idx}.png"), seg_mask_mono)

            # # generate rectified occlusion map using both sfm and stereo point clouds
            # cam_extrinsics = bev_generator.get_updated_camera_extrinsics()
            # cam_extrinsics = cam_extrinsics[:3, :3]
            # cam_extrinsics_inv = np.linalg.inv(cam_extrinsics)
            

            # save rectified-sfm point cloud
            # sfm_pcd_rectified = sfm_pcd.clone()
            # sfm_pcd_rectified.rotate(cam_extrinsics_inv, center=(0, 0, 0))
            # o3d.t.io.write_point_cloud(str(output_dirs["rectified_sfm_pcd"] / f"rectified-sfm-pcd-{idx}.ply"), sfm_pcd_rectified)

            # logger.warning(f"───────────────────────────────")
            # logger.warning(f"len(sfm_pcd): {len(sfm_pcd.point['positions'].numpy())}")
            # logger.warning(f"len(sfm_pcd_rectified): {len(sfm_pcd_rectified.point['positions'].numpy())}")
            # logger.warning(f"───────────────────────────────\n")
            
            # fill nan with 0
            # stereo_pcd_0 = OccMap.get_stereo_pcd(left_img, right_img,
            #                                   K=camera_matrix,
            #                                   baseline=0.12,
            #                                   fill_nan_value=0)

            # stereo_pcd_0 = OccMap.get_stereo_pcd(left_img, right_img,
            #                                   K=camera_matrix,
            #                                   baseline=0.12,
            #                                   fill_nan_value=1)


            # # fill nan with 120
            # stereo_pcd_120 = OccMap.get_stereo_pcd(left_img, right_img,
            #                                   K=camera_matrix,
            #                                   baseline=0.12,
            #                                   fill_nan_value=120)



            # o3d.t.io.write_point_cloud(str(output_dirs["stereo_pcd"] / f"stereo-pcd-{idx}.ply"), stereo_pcd_0)
            
            # # combine pcds
            # combined_pcd = OccMap.combine_pcds([stereo_pcd_0, sfm_pcd])
            # o3d.t.io.write_point_cloud(str(output_dirs["combined_pcd"] / f"combined-pcd-{idx}.ply"), combined_pcd)


            # rectified stereo pcd
            # stereo_pcd_rectified = stereo_pcd_0.clone()
            # stereo_pcd_rectified.rotate(cam_extrinsics_inv, center=(0, 0, 0))
            # o3d.t.io.write_point_cloud(str(output_dirs["rectified_stereo_pcd"] / f"rectified-stereo-pcd-{idx}.ply"), stereo_pcd_rectified)

            # logger.warning(f"───────────────────────────────")
            # logger.warning(f"len(stereo_pcd_rectified): {len(stereo_pcd_rectified.point['positions'].numpy())}")
            # logger.warning(f"len(stereo_pcd_0): {len(stereo_pcd_0.point['positions'].numpy())}")
            # logger.warning(f"len(stereo_pcd_120): {len(stereo_pcd_120.point['positions'].numpy())}")
            # logger.warning(f"───────────────────────────────\n")


            # # combined rectified pcds
            # combined_rectified_pcd = OccMap.combine_pcds([stereo_pcd_rectified, sfm_pcd_rectified])
            # o3d.t.io.write_point_cloud(str(output_dirs["combined_rectified_pcd"] / f"combined-rectified-pcd-{idx}.ply"), combined_rectified_pcd)


            # # save stereo / sfm pcd projections
            # stereo_img_0 = OccMap.project_pcd_to_img(stereo_pcd_0,
            #                            K=camera_matrix,
            #                            img_shape = (1080, 1920),
            #                            visualize=False)
            
            # sfm_img = OccMap.project_pcd_to_img(sfm_pcd,
            #                            K=camera_matrix,
            #                            img_shape = (1080, 1920),
            #                            visualize=False)
            
            # cv2.imwrite(str(output_dirs["stereo_img"] / f"stereo-img-{idx}.jpg"), stereo_img_0)
            # cv2.imwrite(str(output_dirs["sfm_img"] / f"sfm-img-{idx}.jpg"), sfm_img)

            

            # # generate and save sfm_occ-pcd
            # # WITH CROP
            # sfm_occ_pcd = OccMap.get_sfm_occ_pcd(
            #     sfm_pcd,
            #     # sfm_pcd_downsampled,
            #     K = camera_matrix, 
            #     to_crop=True,
            #     # bb={'x_min': -2.49, 'x_max': 2.49, 'z_min': 0.02, 'z_max': 5},
            #     bb={'x_min': -2.5, 'x_max': 2.5, 'z_min': 2.0, 'z_max': 7.0},
            #     img_shape=(1080, 1920))
            
            # NO CROP
            # sfm_occ_pcd = OccMap.get_sfm_occ_pcd(
            #     sfm_pcd,
            #     K = camera_matrix, 
            #     to_crop=False,
            #     # bb={'x_min': -2.49, 'x_max': 2.49, 'z_min': 0.02, 'z_max': 5},
            #     img_shape=(1080, 1920))
            
            # o3d.t.io.write_point_cloud(
            #     str(output_dirs["sfm_occ_pcd"] / f"sfm-occ-pcd-{idx}.ply"),
            #     sfm_occ_pcd)
            
            # logger.info(f"───────────────────────────────")
            # logger.info(f"Generating stereo_occ_pcd map for {idx}")
            # logger.info(f"───────────────────────────────")

            # # stereo_occ_pcd
            # P1 = np.hstack([np.eye(3), np.zeros((3,1))])
            # stereo_occ_pcd = OccMap.get_stereo_occ_pcd(
            #     sfm_pcd=sfm_pcd,
            #     # sfm_pcd=sfm_pcd_downsampled,
            #     stereo_pcd=stereo_pcd_120,
            #     K=camera_matrix,
            #     P=P1,
            #     to_crop=False,
            #     bb={'x_min': -2.5, 'x_max': 2.5, 'z_min': 0.0, 'z_max': 5.0},
            #     img_shape=(1080, 1920)
            # )
            
            # o3d.t.io.write_point_cloud(
            #     str(output_dirs["stereo_occ_pcd"] / f"stereo-occ-pcd-{idx}.ply"),
            #     stereo_occ_pcd
            # )

            # logger.info(f"───────────────────────────────")
            # logger.info(f"Generating rectified stereo_occ_pcd map for {idx}")
            # logger.info(f"───────────────────────────────")

          
           
            # # rectified_stereo_occ_pcd
            # P2 = np.hstack([cam_extrinsics, np.zeros((3,1))])
            # rectified_stereo_occ_pcd = OccMap.get_stereo_occ_pcd(
            #     sfm_pcd=sfm_pcd_rectified,
            #     stereo_pcd=stereo_pcd_rectified,
            #     K=camera_matrix,
            #     P=P2,
            #     to_crop=False,
            #     bb={'x_min': -2.5, 'x_max': 2.5, 'z_min': 0.0, 'z_max': 10.0},
            #     img_shape=(1080, 1920)
            # )
            
            # o3d.t.io.write_point_cloud(
            #     str(output_dirs["rectified_stereo_occ_pcd"] / f"rectified-stereo-occ-pcd-{idx}.ply"),
            #     rectified_stereo_occ_pcd
            # )


            
            # combine pcds
            
            # # red for stereo pcd
            # colored_stereo_pcd = OccMap.color_pcd(stereo_pcd_0, color=[255, 0, 0])
            
            # # yellow for sfm pcd
            # colored_sfm_pcd = OccMap.color_pcd(sfm_pcd, color=[255, 255, 0])

            # combined_pcd = OccMap.combine_pcds([colored_stereo_pcd, colored_sfm_pcd])
            # o3d.t.io.write_point_cloud(str(output_dirs["combined_pcd"] / f"combined-pcd-{idx}.ply"), combined_pcd)
            
            
            # cam_extrinsics = bev_generator.get_updated_camera_extrinsics()
            # cam_extrinsics = cam_extrinsics[:3, :3]
            # cam_extrinsics_inv = np.linalg.inv(cam_extrinsics)

            # # rectified stereo-occ-pcd
            # rectified_stereo_occ_pcd = stereo_occ_pcd.clone()
            # rectified_stereo_occ_pcd.rotate(cam_extrinsics_inv, center=(0, 0, 0))
            # o3d.t.io.write_point_cloud(str(output_dirs["rectified_stereo_occ_pcd"] / f"rectified-stereo-occ-pcd-{idx}.ply"), rectified_stereo_occ_pcd)



        except Exception as e:
            logger.error(f"Error processing {pcd_files[idx]}: {str(e)}")
            logger.error(traceback.format_exc())

def test_stereo_pcd():
    """Case 11: Test stereo point cloud generation"""
    pcd_dir = Path("debug/frames-4")
    output_dir = Path("debug/4")
    
    output_dirs = {
        "seg_masks": output_dir / "seg-masks",
        "rectified_pcd": output_dir / "rectified-pcd",
        "left_img": output_dir / "left-imgs",
        "right_img": output_dir / "right-imgs",
        "stereo_pcd": output_dir / "stereo-pcd",
        "combined_pcd": output_dir / "combined-pcd",
        "sfm_pcd": output_dir / "sfm-pcd"
    }
    
    for dir_path in output_dirs.values():
        # logger.info("───────────────────────────────")
        # logger.info(f"Creating directory: {dir_path}")
        # logger.info("───────────────────────────────")
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Find all PCD files
    pcd_files = []
    for root, _, files in os.walk(pcd_dir):
        for file in files:
            if file == "left-segmented-labelled.ply":
                pcd_files.append(Path(root) / file)
    
    pcd_files.sort()
    
    
    # Process each PCD file
    for idx, pcd_path in enumerate(tqdm(pcd_files, desc="Processing point clouds")):
        try:

            # if idx > 1:
            #     break
            
            # Read and process images
            img_dir = pcd_path.parent
            left_src = img_dir / "left.jpg"
            right_src = img_dir / "right.jpg"
            
            left_img = cv2.imread(str(left_src))
            right_img = cv2.imread(str(right_src))

            # h, w, _ = left_img.shape
            # left_img_cropped = left_img[h//2:, :w//2]
            # right_img_cropped = right_img[h//2:, :w//2]

            # cv2.imshow("left", left_img_cropped)
            # cv2.imshow("right", right_img_cropped)
            # key = cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # if key == ord('q'):
            #     logger.info("Exiting debug case...")
            #     break

            # left_img = cv2.resize(left_img, (480, 640))
            # right_img = cv2.resize(right_img, (480, 640))
            
            # logger.info(f"Processing images - left: {left_img.shape}, right: {right_img.shape}")
            
            # is_rectified, avg_vertical_disparity = OccMap.is_rectified(left_img, right_img)
            
            # Save processed images
            left_dest = output_dirs["left_img"] / f"left-img-{idx}.jpg"
            right_dest = output_dirs["right_img"] / f"right-img-{idx}.jpg"
            
            cv2.imwrite(str(left_dest), left_img)
            cv2.imwrite(str(right_dest), right_img)
            
            # sfm pointcloud
            sfm_pcd_path = pcd_path
            sfm_pcd = o3d.t.io.read_point_cloud(str(sfm_pcd_path))
            o3d.t.io.write_point_cloud(str(output_dirs["sfm_pcd"] / f"sfm-pcd-{idx}.ply"), sfm_pcd)

            # stereo-pcd calculations
            camera_matrix = np.array([[1090.536, 0, 954.99],
                                   [0, 1090.536, 523.12],
                                   [0, 0, 1]], dtype=np.float32)
            
            
            stereo_pcd = OccMap.get_stereo_pcd(left_img, right_img, 
                                               K = camera_matrix, 
                                               baseline = 0.12)

            o3d.t.io.write_point_cloud(str(output_dirs["stereo_pcd"] / f"stereo-pcd-{idx}.ply"), stereo_pcd)

            # combine pcds
            # sfm_pcd = OccMap.color_pcd(sfm_pcd, color=[255, 0, 255])
            # combined_pcd = OccMap.combine_pcds(sfm_pcd, stereo_pcd, use_distinct_colors=False)
            combined_pcd = OccMap.combine_pcds(stereo_pcd, sfm_pcd, use_distinct_colors=False)
            
            o3d.t.io.write_point_cloud(str(output_dirs["combined_pcd"] / f"combined-pcd-{idx}.ply"), combined_pcd)


        except Exception as e:
            logger.error(f"Error processing {pcd_path}: {str(e)}")
            logger.error(traceback.format_exc())

def project_pcd_to_camera(pcd_input, camera_matrix, image_size, rvec=None, tvec=None):
    """Project pointcloud to camera view"""
    assert rvec is not None, "rvec must be provided"
    assert tvec is not None, "tvec must be provided"
    
    points = pcd_input.point['positions'].numpy()
    colors = pcd_input.point['colors'].numpy()
    
    projected_points, _ = cv2.projectPoints(points, rvec=rvec, tvec=tvec, 
                                          cameraMatrix=camera_matrix, distCoeffs=None)
    projected_points = projected_points.reshape(-1, 2)
    
    img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    
    for idx, (point, color) in enumerate(zip(projected_points, colors)):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            color_bgr = (int(color[2]), int(color[1]), int(color[0]))
            cv2.circle(img, (x, y), 3, color_bgr, -1)
    
    return img

def plot_segmentation_classes(mask: np.ndarray, path: str = None, title: str = None) -> None:
    """Plot segmentation classes with their colors"""
    unique_classes = np.unique(mask)
    plt.figure(figsize=(10, 6))
    
    with open("config/Mavis.yaml", 'r') as f:
        config = yaml.safe_load(f)
    color_map = config['color_map']
    
    for class_id in unique_classes:
        y_coords, x_coords = np.where(mask == class_id)
        color = color_map.get(class_id, [0, 0, 0])
        color = [c/255 for c in color[::-1]]
        plt.scatter(x_coords, y_coords, label=f'Class {class_id}', color=color, alpha=0.5)
    
    if title:
        plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
    else:
        plt.show()
    plt.close()

def test_aws_occ_generation():
    """Case 10: Test occlusion map generation [aws version]"""
    pcd_dir = Path("debug/frames-4")
    output_dirs = {
        "seg_masks": pcd_dir / "seg-masks",
        "rectified_pcd": pcd_dir / "rectified-pcd",
        "left_img": pcd_dir / "left-imgs",
        "right_img": pcd_dir / "right-imgs",
        "occ_pcd": pcd_dir / "occ-pcd"
    }
    
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    OCC_FOLDER_URI = ["s3://occupancy-dataset/occ-dataset/vineyards/gallo/"]
    data_generator_s3 = DataGeneratorS3(
        src_URIs=OCC_FOLDER_URI,
        dest_folder=None,
        index_json="dummy",
        color_map="dummy",
        crop_bb={"x": 0, "y": 0, "w": 1, "h": 1, "d": 1},
        nx=1,
        nz=1
    )
    
    leaf_URIs = data_generator_s3.get_leaf_folders()
    
    for idx, leaf_URI in enumerate(leaf_URIs):
        try:
            logger.info(f"Processing leaf URI: {leaf_URI}")
            
            leaf_folder = LeafFolder(
                src_URI=leaf_URI,
                dest_URI="dummy_dest",
                index_json="dummy_index.json",
                crop_bb={"x": 0, "y": 0, "z": 0, "w": 1, "h": 1, "d": 1},
                color_map="dummy_color_map",
                nx=1,
                nz=1
            )
            
            # Process PCD and images
            # [Add implementation details from debug.py Case 10]
            
        except Exception as e:
            logger.error(f"Error processing {leaf_URI}: {str(e)}")
            logger.error(traceback.format_exc())


def test_occ_generation():
    """Case 8: Test occlusion map generation"""
    pcd_dir = Path("debug/frames-4")
    output_dirs = {
        "seg_masks": pcd_dir / "seg-masks",
        "rectified_pcd": pcd_dir / "rectified-pcd",
        "left_img": pcd_dir / "left-imgs",
        "right_img": pcd_dir / "right-imgs",
        "occ_pcd": pcd_dir / "occ-pcd"
    }
    
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Find all PCD files
    pcd_files = []
    for root, _, files in os.walk(pcd_dir):
        for file in files:
            if file == "left-segmented-labelled.ply":
                pcd_files.append(Path(root) / file)
    
    pcd_files.sort()
    
    # Initialize generators
    bev_generator = BEVGenerator(logging_level=logging.WARNING)
    
    # Process each PCD file
    for idx, pcd_path in enumerate(tqdm(pcd_files, desc="Processing point clouds")):
        try:
            # Read and process images
            img_dir = pcd_path.parent
            left_src = img_dir / "left.jpg"
            right_src = img_dir / "right.jpg"
            
            left_img = cv2.imread(str(left_src))
            right_img = cv2.imread(str(right_src))
            
            left_img = cv2.resize(left_img, (480, 640))
            right_img = cv2.resize(right_img, (480, 640))
            
            logger.info(f"Processing images - left: {left_img.shape}, right: {right_img.shape}")
            
            is_rectified, avg_vertical_disparity = OccMap.is_rectified(left_img, right_img)
            
            # Save processed images
            left_dest = output_dirs["left_img"] / f"left-img-{idx}.jpg"
            right_dest = output_dirs["right_img"] / f"right-img-{idx}.jpg"
            
            cv2.imwrite(str(left_dest), left_img)
            cv2.imwrite(str(right_dest), right_img)
            
            # Process point cloud
            pcd_input = o3d.t.io.read_point_cloud(str(pcd_path))
            
            # Generate and save segmentation masks
            crop_bb = {'x_min': -2.49, 'x_max': 2.49, 'z_min': 0.02, 'z_max': 5}
            seg_mask_mono, seg_mask_rgb = bev_generator.pcd_to_seg_mask(
                pcd_input, nx=256, nz=256, bb=crop_bb, yaml_path="config/Mavis.yaml"
            )
            
            cv2.imwrite(str(output_dirs["seg_masks"] / f"seg-mask-rgb-{idx}.png"), seg_mask_rgb)
            
            # Generate and save rectified point cloud
            pcd_rectified = bev_generator.get_tilt_rectified_pcd(pcd_input)
            o3d.t.io.write_point_cloud(
                str(output_dirs["rectified_pcd"] / f"rectified-pcd-{idx}.ply"), 
                pcd_rectified
            )
            
            # Generate occlusion map
            camera_matrix = np.array([[1090.536, 0, 954.99],
                                   [0, 1090.536, 523.12],
                                   [0, 0, 1]], dtype=np.float32)
            
            # camera_projection_matrix = bev_generator.get_updated_camera_extrinsics(pcd_input)
            pcd_downsampled = bev_generator.get_downsampled_pcd()
            
            occ_pcd = OccMap.get_occ_mask(
                pcd=pcd_downsampled,
                camera_matrix=camera_matrix,
                bb=crop_bb,
                camera_projection_matrix=camera_projection_matrix
            )
            
            o3d.t.io.write_point_cloud(
                str(output_dirs["occ_pcd"] / f"occ-pcd-{idx}.ply"),
                occ_pcd
            )
            
        except Exception as e:
            logger.error(f"Error processing {pcd_path}: {str(e)}")
            logger.error(traceback.format_exc())

def test_image_size():
    """Case 7: Check image size"""
    img_path = "data/train-data/0/_left.jpg"
    image = cv2.imread(img_path)
    if image is not None:
        height, width, _ = image.shape
        logger.info(f"Image size: {width}x{height}")
    else:
        logger.error("Failed to read the image")

def test_camera_extrinsics():
    """Case 6: Testing bev_generator.updated_camera_extrinsics()"""
    bev_generator = BEVGenerator()
    ground_id = bev_generator.LABELS["NAVIGABLE_SPACE"]["id"]
    camera_matrix = np.array([[1093.2768, 0, 964.989],
                           [0, 1093.2768, 569.276],
                           [0, 0, 1]], dtype=np.float32)
    
    pcd_input = o3d.t.io.read_point_cloud("debug/left-segmented-labelled.ply")
    pcd_rectified = bev_generator.tilt_rectification(pcd_input)
    
    R = bev_generator.compute_tilt_matrix(pcd_input)
    yaw_i, pitch_i, roll_i = bev_generator.rotation_matrix_to_ypr(R)
    
    logger.info(f"Initial angles - yaw: {yaw_i}, pitch: {pitch_i}, roll: {roll_i}")
    
    is_orthogonal = np.allclose(np.dot(R.T, R), np.eye(3), atol=1e-6)
    logger.info(f"Is rotation matrix orthogonal? {is_orthogonal}")
    
    R_transpose = R.T
    yaw_f, pitch_f, roll_f = bev_generator.rotation_matrix_to_ypr(R_transpose)
    logger.info(f"Final angles - yaw: {yaw_f}, pitch: {pitch_f}, roll: {roll_f}")
    
    # Test projection
    img_i = project_pcd_to_camera(pcd_input, camera_matrix, (1920, 1080), 
                                 rvec=np.zeros((3, 1)), tvec=np.zeros((3, 1)))
    
    rvec, _ = cv2.Rodrigues(R_transpose)
    img_f = project_pcd_to_camera(pcd_rectified, camera_matrix, (1920, 1080), 
                                 rvec=rvec, tvec=np.zeros((3, 1)))

def test_bev_generator():
    """Case 5: Testing BEVGenerator"""
    pcd_input = o3d.t.io.read_point_cloud("debug/frames/frame-2686/left-segmented-labelled.ply")
    bev_generator = BEVGenerator()
    pcd_rectified = bev_generator.tilt_rectification(pcd_input)
    logger.info("BEV generation test completed")

def test_camera_projection():
    """Test pointcloud to camera projection (Case 4)"""
    from scripts.helpers import read_images_binary, cam_extrinsics
    
    images = read_images_binary("debug/dense-reconstruction/images.bin")
    sparse_image_keys = sorted(images.keys())
    num_sparse_images = len(sparse_image_keys)
    
    logger.info(f"Found {num_sparse_images} sparse images")
    
    # Use image 76 as example
    left_img = images[sparse_image_keys[76]]
    logger.info(f"Processing image: {left_img.name}")
    
    # Get camera extrinsics
    C_l = cam_extrinsics(left_img)
    pcd_input = o3d.t.io.read_point_cloud("debug/dense-reconstruction/dense.ply")
    
    # Camera parameters
    tvec_i = C_l[:3, 3]
    rvec_i = cv2.Rodrigues(C_l[:3, :3])[0]
    camera_matrix = np.array([[1090.536, 0, 954.99],
                           [0, 1090.536, 523.12],
                           [0, 0, 1]], dtype=np.float32)
    
    # Project points and display
    img = project_pcd_to_camera(pcd_input, camera_matrix, (1920, 1080), 
                               rvec=rvec_i, tvec=tvec_i)
    
    cv2.imshow("Projected Points", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def test_camera_params():
    """Test camera parameters (Case 2)"""
    from scripts.helpers import get_zed_camera_params
    
    camera_params = get_zed_camera_params("debug/front_2024-06-05-09-14-54.svo")
    logger.info(f"Camera parameters: {camera_params}")

def test_segmentation_masks(case_1=True):
    """Case 1: Generate segmentation masks"""
    vis = o3d.visualization.Visualizer()
    bev_generator = BEVGenerator()
    
    pcd_input = o3d.t.io.read_point_cloud("debug/left-segmented-labelled.ply")
    pcd_rectified = bev_generator.tilt_rectification(pcd_input)
    
    ground_id = bev_generator.LABELS["NAVIGABLE_SPACE"]["id"]
    
    # Ground plane normal => [original / rectified] pointcloud
    n_i, _ = bev_generator.get_class_plane(pcd_input, ground_id)
    n_f, _ = bev_generator.get_class_plane(pcd_rectified, ground_id)
    
    # Pitch, yaw, roll => [original / rectified]
    p_i, y_i, r_i = bev_generator.axis_angles(n_i)
    p_f, y_f, r_f = bev_generator.axis_angles(n_f)

    logger.info(f"[BEFORE RECTIFICATION] - Yaw: {y_i:.2f}, Pitch: {p_i:.2f}, Roll: {r_i:.2f}")
    logger.info(f"[AFTER RECTIFICATION] - Yaw: {y_f:.2f}, Pitch: {p_f:.2f}, Roll: {r_f:.2f}")

    # Generate BEV
    bev_pcd = bev_generator.generate_BEV(pcd_input)
    logger.info(f"Number of points in bev_pcd: {len(bev_pcd.point['positions'].numpy())}")
    
    # Cropping params
    crop_bb = {'x_min': -5, 'x_max': 5, 'z_min': 0, 'z_max': 10}
    bev_pcd_cropped = crop_pcd(bev_pcd, crop_bb)

    # Generate and save segmentation mask
    seg_mask_mono = bev_generator.bev_to_seg_mask_mono(
        bev_pcd_cropped, nx=400, nz=400, bb=crop_bb
    )
    seg_mask_rgb = mono_to_rgb_mask(seg_mask_mono)
    
    output_path = "debug/seg-mask-rgb.png"
    cv2.imwrite(output_path, seg_mask_rgb)
    logger.info(f"Segmentation mask saved to {output_path}")

    if case_1:
        # Visualization
        vis.create_window()
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)
        vis.add_geometry(bev_pcd_cropped.to_legacy())
        
        view_ctr = vis.get_view_control()
        view_ctr.set_front(np.array([0, -1, 0]))
        view_ctr.set_up(np.array([0, 0, 1]))
        view_ctr.set_zoom(4)
        
        vis.run()
        vis.destroy_window()

def test_bev_generation():
    """Test basic BEV generation functionality (Case 1)"""
    src_folder = "train-data"
    dst_folder = "debug/output-seg-masks"
    bev_generator = BEVGenerator()

    crop_bb = {'x_min': -5, 'x_max': 5, 'z_min': 0, 'z_max': 10}
    nx = 400
    nz = 400

    os.makedirs(dst_folder, exist_ok=True)
    
    left_segmented_labelled_files = []
    total_files = sum(len(files) for _, _, files in os.walk(src_folder) if 'left-segmented-labelled.ply' in files)
    
    with tqdm(total=total_files, desc="Processing files", ncols=100) as pbar:
        for root, _, files in os.walk(src_folder):
            for file in files:
                if file == 'left-segmented-labelled.ply':
                    file_path = os.path.join(root, file)
                    left_segmented_labelled_files.append(file_path)

                    try:
                        pcd_input = o3d.t.io.read_point_cloud(file_path)
                        seg_mask_mono, seg_mask_rgb = bev_generator.pcd_to_seg_mask(
                            pcd_input, nx=nx, nz=nz, bb=crop_bb
                        )
                        output_rgb_path = os.path.join(dst_folder, f"seg-mask-rgb-{os.path.basename(root)}.png")
                        cv2.imwrite(output_rgb_path, seg_mask_rgb)
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                    pbar.update(1)
