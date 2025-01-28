#! /usr/bin/env python3

import boto3
from urllib.parse import urlparse
import open3d as o3d
import yaml
import numpy as np
from pathlib import Path

from logger import get_logger


def is_close(colors: np.ndarray, target_color: tuple, threshold: int = 20) -> np.ndarray:
    """
    Check if colors are close to target_color by comparing each RGB component individually.

    Args:
        colors (np.ndarray): A numpy array of shape (N, 3) containing RGB colors.
        target_color (tuple): A tuple of (R, G, B) values to compare against.
        threshold (int): Maximum difference for each color component to be considered close.

    Returns:
        np.ndarray: A boolean mask array of shape (N,) where True indicates close colors.
    """
    # Convert target_color to a numpy array for broadcasting
    target_color = np.array(target_color)
    
    # Calculate the absolute difference for each color component
    color_diff = np.abs(colors - target_color)
    
    # Check if all color components are within the threshold
    return np.all(color_diff <= threshold, axis=1)

def correct_pcd_label(pcd: o3d.t.geometry.PointCloud, dairy_yaml: str) -> o3d.t.geometry.PointCloud:
    """
    correct point cloud labels based on point colors using the mapping in dairy.yaml
    
    args:
        pcd: point cloud object
        dairy_yaml: path to the yaml file containing color to label mapping
    """
    logger = get_logger("correct_pcd_label")
    
    with open(dairy_yaml, 'r') as file:
        dairy_config = yaml.safe_load(file)
    
    # get colors and labels from yaml
    color_to_label = {}
    for label, color in dairy_config['color_map'].items():
        # convert yaml BGR colors to RGB for comparison
        rgb_color = (color[2], color[1], color[0])  # BGR to RGB
        color_to_label[tuple(rgb_color)] = label
    
    # convert pcd colors and labels to numpy for efficient operations
    colors = pcd.point['colors'].numpy()  # shape: (N, 3), in RGB
    labels = pcd.point['label'].numpy()  # shape: (N,)
    
    # scale colors to match yaml values (0-255)
    colors = colors.astype(np.int32)
    
    # create a mask for all points that have not been labeled yet
    unlabeled_mask = np.ones(len(labels), dtype=bool)

    # iterate through each defined color and update matching points
    for target_color, label in color_to_label.items():
        # update labels for points with similar colors
        similar_color_mask = is_close(colors, target_color, threshold=20)
        labels[similar_color_mask] = label
        # update the unlabeled mask
        unlabeled_mask[similar_color_mask] = False
    
    # set the label for points not in color_to_label to 0
    labels[unlabeled_mask] = 0
    
    # update pcd labels
    pcd.point['label'] = o3d.core.Tensor(labels)
    
    # save corrected pcd
    return pcd

def get_s3_client():
    """Create and return an S3 client"""
    return boto3.client('s3')

def parse_s3_uri(s3_uri):
    """Parse S3 URI into bucket and prefix"""
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/')
    return bucket, prefix

def list_numbered_folders(s3_uri):
    """List all folders that match the pattern xxxx_to_xxxx"""
    s3_client = get_s3_client()
    bucket, prefix = parse_s3_uri(s3_uri)
    
    # List all objects with the given prefix
    paginator = s3_client.get_paginator('list_objects_v2')
    matching_folders = set()
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            # Split the path into components
            parts = key.split('/')
            
            # Look for folders matching the pattern xxxx_to_xxxx
            for part in parts:
                if '_to_' in part and part[0].isdigit():
                    folder_path = f"s3://{bucket}/{key[:key.index(part) + len(part)]}"
                    matching_folders.add(folder_path)
    
    return sorted(list(matching_folders))

def inspect_pcd_labels(pcd: o3d.t.geometry.PointCloud):
    """ inspect point cloud labels"""

    logger = get_logger("inspect_pcd_labels")

    pcd_labels = pcd.point['label'].numpy()

    logger.info("───────────────────────────────")
    logger.info(f"pcd_labels.shape: {pcd_labels.shape}")
    logger.info(f"pcd_labels.dtype: {pcd_labels.dtype}")
    logger.info("───────────────────────────────")
    
    labels = [0,1,2,3,5,10,11,13]

    logger.info("───────────────────────────────")
    
    for label in labels:
        logger.info(f"label: {label}")
        label_mask = pcd_labels == label
        
        valid_labels = pcd.point['label'].numpy()[label_mask]

        logger.info(f"label: {label}, count: {len(valid_labels)} percent: {len(valid_labels) / len(pcd_labels) * 100:.2f}%")

    logger.info("───────────────────────────────")

if __name__ == "__main__":
    logger = get_logger("update_labels")
    base_uri = "s3://occupancy-dataset/occ-dataset/dairy/"
    
    # try:
    #     folders = list_numbered_folders(base_uri)
        
    #     # logger.info("───────────────────────────────")
    #     # for folder in folders[0:1]:
    #     #     logger.info(f"folder: {folder}")
    #     # logger.info("───────────────────────────────")

    

    # except Exception as e:
    #     print(f"Error: {str(e)}")

    pcd_path = Path("debug/frames-16/frame-920/left-segmented-labelled.ply")
    dairy_yaml = Path("config/dairy.yaml")

    logger.warning("───────────────────────────────")
    logger.warning("BEFORE CORRECTING LABELS")
    logger.warning("───────────────────────────────")
    
    pcd = o3d.t.io.read_point_cloud(pcd_path.as_posix())
    inspect_pcd_labels(pcd)    
    
    
    updated_pcd = correct_pcd_label(pcd, dairy_yaml.as_posix())
    
    logger.warning("───────────────────────────────")
    logger.warning("AFTER CORRECTING LABELS")
    logger.warning("───────────────────────────────")

    
    logger.info("───────────────────────────────")
    logger.info("updated_pcd.point['label'].numpy().shape: %s", updated_pcd.point['label'].numpy().shape)
    logger.info("updated_pcd.point['colors'].numpy().shape: %s", updated_pcd.point['colors'].numpy().shape)
    logger.info("updated_pcd.point['positions'].numpy().shape: %s", updated_pcd.point['positions'].numpy().shape)
    logger.info("───────────────────────────────")

    inspect_pcd_labels(updated_pcd)

