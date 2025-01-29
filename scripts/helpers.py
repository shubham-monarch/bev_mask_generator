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

from scripts.logger import get_logger


logger = get_logger("helpers")

def list_base_folders(folder_path):
    base_folders = []
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            base_folders.append(os.path.join(root, dir_name))
    return base_folders


def get_label_colors_from_yaml(yaml_path=None):
    """Read label colors from Mavis.yaml config file."""
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Get BGR colors directly from yaml color_map
    label_colors_bgr = config['color_map']
    
    # Convert BGR to RGB by reversing color channels
    label_colors_rgb = {
        label: color[::-1] 
        for label, color in label_colors_bgr.items()
    }
    
    return label_colors_bgr, label_colors_rgb
        

def mono_to_rgb_mask(mono_mask: np.ndarray, yaml_path: str = None) -> np.ndarray:
    """Convert single channel segmentation mask to RGB using label mapping from a YAML file."""
    
    assert yaml_path is not None, "yaml_path is required!"
    label_colors_bgr, _ = get_label_colors_from_yaml(yaml_path)
    
    H, W = mono_mask.shape
    rgb_mask = np.zeros((H, W, 3), dtype=np.uint8)
    
    for label_id, rgb_value in label_colors_bgr.items():
        mask = mono_mask == label_id
        rgb_mask[mask] = rgb_value
        
    return rgb_mask


def get_zed_camera_params(svo_loc):
    
    zed_file_path = os.path.abspath(svo_loc)
    print(f"zed_file_path: {zed_file_path}")

    input_type = sl.InputType()
    input_type.set_from_svo_file(zed_file_path)
    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    init.coordinate_units = sl.UNIT.METER   

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()


    calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
    zed_camera_params = [calibration_params.left_cam.fx, calibration_params.left_cam.fy, calibration_params.left_cam.cx, calibration_params.left_cam.cy, 0, 0 ,0 , 0]
    
    #print(f"zed_camera_params: {zed_camera_params}")
    camera_params= ",".join(str(x) for x in zed_camera_params)

    return camera_params

def count_unique_labels(mask_img: np.ndarray):
    """Count and return the number of unique labels in a segmented mask image."""
    
    if mask_img.ndim == 3:
        # Convert RGB to single integer for each unique color
        mask_flat = mask_img.reshape(-1, 3)
    elif mask_img.ndim == 2:
        # For 2D array, flatten directly
        mask_flat = mask_img.flatten()
    else:
        raise ValueError("mask_img must be either a 2D or 3D array")
    
    unique_colors = np.unique(mask_flat, axis=0)
    
    return len(unique_colors), unique_colors


def crop_pcd(pcd: o3d.t.geometry.PointCloud, bb: dict = None) -> o3d.t.geometry.PointCloud:
    """Crop a point cloud to a specified area."""
    
    valid_indices = np.where(

        # x between x_min and x_max
        (pcd.point['positions'][:, 0].numpy() >= bb['x_min']) & 
        (pcd.point['positions'][:, 0].numpy() <= bb['x_max']) & 

        # z between z_min and z_max
        (pcd.point['positions'][:, 2].numpy() >= bb['z_min']) & 
        (pcd.point['positions'][:, 2].numpy() <= bb['z_max'])

    )[0]

    return pcd.select_by_index(valid_indices)
