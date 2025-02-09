#!/usr/bin/env python3

import os
import shutil
import open3d as o3d
from tqdm import tqdm
import numpy as np
import cv2
import os
import fnmatch
import logging
import sys
import coloredlogs
import yaml
import pyzed.sl as sl
import matplotlib.pyplot as plt
import fire
from typing import List, Tuple, Dict, Optional

from scripts.logger import get_logger


logger = get_logger("helpers")

def list_base_folders(folder_path: str) -> List[str]:
    """
    list base folders in the given folder path.
    
    :param folder_path: the path to the folder
    :return: list of base folder paths
    """
    base_folders: List[str] = []
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            base_folders.append(os.path.join(root, dir_name))
    return base_folders


def get_label_colors_from_yaml(yaml_path: Optional[str] = None) -> Tuple[dict, dict]:
    """
    read label colors from a YAML config file.

    :param yaml_path: path to the yaml file, required.
    :return: tuple of (label_colors_bgr, label_colors_rgb)
    """
    if yaml_path is None:
        raise ValueError("yaml_path is required!")
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error("failed to read yaml file", exc_info=True)
        raise e
        
    # get bgr colors directly from yaml color_map
    label_colors_bgr = config['color_map']
    
    # convert bgr to rgb by reversing color channels
    label_colors_rgb = {
        label: color[::-1] 
        for label, color in label_colors_bgr.items()
    }
    
    return label_colors_bgr, label_colors_rgb


def mono_to_rgb_mask(mono_mask: np.ndarray, yaml_path: str) -> np.ndarray:
    """
    convert single channel segmentation mask to rgb using label mapping from a yaml file.

    :param mono_mask: input single channel mask as numpy array.
    :param yaml_path: yaml file containing label-color mapping.
    :return: rgb mask as numpy array.
    """
    if yaml_path is None:
        raise ValueError("yaml_path is required!")
        
    label_colors_bgr, _ = get_label_colors_from_yaml(yaml_path)
    
    H, W = mono_mask.shape
    rgb_mask = np.zeros((H, W, 3), dtype=np.uint8)
    
    for label_id, rgb_value in label_colors_bgr.items():
        mask = mono_mask == label_id
        rgb_mask[mask] = rgb_value
        
    return rgb_mask


def get_zed_camera_params(svo_loc: str) -> str:
    """
    get zed camera calibration parameters from a given svo file.

    :param svo_loc: path to the svo file.
    :return: comma separated string of camera parameters.
    """
    try:
        zed_file_path = os.path.abspath(svo_loc)
        logger.info(f"zed_file_path: {zed_file_path}")
    
        input_type = sl.InputType()
        input_type.set_from_svo_file(zed_file_path)
        init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
        init.coordinate_units = sl.UNIT.METER   
    
        zed = sl.Camera()
        status = zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            logger.error(f"failed to open zed camera: {repr(status)}")
            sys.exit(1)
    
        calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
        zed_camera_params = [
            calibration_params.left_cam.fx, 
            calibration_params.left_cam.fy, 
            calibration_params.left_cam.cx, 
            calibration_params.left_cam.cy, 
            0, 0, 0, 0
        ]
        # removed debug print statement for cleaner logs
        camera_params = ",".join(str(x) for x in zed_camera_params)
    
        return camera_params
    except Exception as error:
        logger.error("error in get_zed_camera_params", exc_info=True)
        raise error


def count_unique_labels(mask_img: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    count and return the number of unique labels in a segmented mask image.

    :param mask_img: segmented mask image as numpy array.
    :return: tuple of (number of unique labels, array of unique colors)
    """
    if mask_img.ndim == 3:
        # convert rgb to single integer for each unique color
        mask_flat = mask_img.reshape(-1, 3)
    elif mask_img.ndim == 2:
        # for 2d array, flatten directly
        mask_flat = mask_img.flatten()
    else:
        raise ValueError("mask_img must be either a 2d or 3d array")
    
    unique_colors = np.unique(mask_flat, axis=0)
    
    return len(unique_colors), unique_colors


def crop_pcd(pcd: o3d.t.geometry.PointCloud, bb: Optional[Dict[str, float]] = None) -> o3d.t.geometry.PointCloud:
    """
    crop a point cloud to a specified area defined by the bounding box.

    :param pcd: the open3d tensor point cloud.
    :param bb: dictionary with keys: 'x_min', 'x_max', 'z_min', 'z_max'.
    :return: cropped point cloud.
    """
    if bb is None:
        raise ValueError("bounding box dictionary 'bb' is required")
    
    valid_indices = np.where(
        # x between x_min and x_max
        (pcd.point['positions'][:, 0].numpy() >= bb['x_min']) & 
        (pcd.point['positions'][:, 0].numpy() <= bb['x_max']) & 
        # z between z_min and z_max
        (pcd.point['positions'][:, 2].numpy() >= bb['z_min']) & 
        (pcd.point['positions'][:, 2].numpy() <= bb['z_max'])
    )[0]

    return pcd.select_by_index(valid_indices)


def show_help() -> None:
    """
    display the help information for the helper functions available via the CLI.

    available commands:
      list_base_folders(folder_path) : list all base subfolders in a given directory path.
      get_label_colors_from_yaml(yaml_path) : retrieve label colors from a yaml configuration file.
      mono_to_rgb_mask(mono_mask, yaml_path) : convert a single channel segmentation mask to an rgb mask.
      get_zed_camera_params(svo_loc) : get camera parameters from a zed camera svo file.
      count_unique_labels(mask_img) : count and return unique labels in a segmented mask image.
      crop_pcd(pcd, bb) : crop a point cloud to a specified area defined by bounding box (bb).
      show_help() : display this help text.
    """
    help_text = (
        "\nhelper functions basic usage:\n\n"
        "1. list_base_folders(folder_path):\n"
        "   list all base subfolders in a given directory path.\n\n"
        "2. get_label_colors_from_yaml(yaml_path):\n"
        "   retrieve label colors from a yaml configuration file.\n\n"
        "3. mono_to_rgb_mask(mono_mask, yaml_path):\n"
        "   convert a single channel segmentation mask to an rgb mask using provided yaml mapping.\n\n"
        "4. get_zed_camera_params(svo_loc):\n"
        "   get camera calibration parameters from a zed camera svo file.\n\n"
        "5. count_unique_labels(mask_img):\n"
        "   count and return the number of unique labels in a segmented mask image.\n\n"
        "6. crop_pcd(pcd, bb):\n"
        "   crop a point cloud to a specified area defined by bounding box (bb).\n\n"
        "7. show_help():\n"
        "   display this help text.\n"
    )
    logger.info(help_text)


def main() -> None:
    """
    main function to provide command-line interface for the helper functions using the Fire module.

    functions available:
      list_base_folders
      get_label_colors_from_yaml
      mono_to_rgb_mask
      get_zed_camera_params
      count_unique_labels
      crop_pcd
      show_help
    """
    try:
        fire.Fire({
            "list_base_folders": list_base_folders,
            "get_label_colors_from_yaml": get_label_colors_from_yaml,
            "mono_to_rgb_mask": mono_to_rgb_mask,
            "get_zed_camera_params": get_zed_camera_params,
            "count_unique_labels": count_unique_labels,
            "crop_pcd": crop_pcd,
            "show_help": show_help
        })
    except Exception as error:
        logger.error("error running main function", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
