'''
Occlusion map generation
'''
#! /usr/bin/env python3

import open3d as o3d
import numpy as np
import open3d.core as o3c
import logging
import cv2
from typing import Dict

from scripts.logger import get_logger
from scripts.helpers import crop_pcd

class OcclusionMap:
    def __init__(self):
        '''Initialize OcclusionMap with default parameters'''
        self.logger = get_logger("occlusion_map", level=logging.INFO)
        
        # Default parameters
        self.focal_length = 500  # pixels
        self.padding = 100  # pixels
        self.depth_threshold = 0.1  # 10cm threshold for occlusion detection
    
    @staticmethod
    def add_occ_mask_to_pcd(pcd: o3d.t.geometry.PointCloud, K: np.ndarray = None) -> o3d.t.geometry.PointCloud:
        '''
        Add occlusion mask to the pointcloud.
        '''
        pass

    # def get_occ_pcd(self, pcd: o3d.t.geometry.PointCloud, K: np.ndarray) -> o3d.t.geometry.PointCloud:
    #     '''
    #     Generate occupancy map from the pointcloud using ray-tracing and depth buffer approach.
    #     Colors occluded points in orange.
        
    #     Args:
    #         pcd (o3d.t.geometry.PointCloud): Input pointcloud in camera frame
    #         K (np.ndarray): 3x3 camera intrinsics matrix
    #             [[fx,  0, cx],
    #              [ 0, fy, cy],
    #              [ 0,  0,  1]]
            
    #     Returns:
    #         o3d.t.geometry.PointCloud: Pointcloud with occluded points colored orange
    #     '''
    #     assert K.shape == (3, 3), "Camera intrinsics must be a 3x3 matrix"
        
    #     # Create a copy of input pointcloud
    #     occ_pcd = pcd.clone()
        
    #     # Get points as numpy array
    #     points = pcd.point['positions'].numpy()
        
    #     self.logger.info(f"================================================")
    #     self.logger.info(f"points.shape: {points.shape}")
    #     self.logger.info(f"================================================\n")

    #     # Calculate depth (distance from camera plane)
    #     depths = points[:, 2]  # z-coordinate is depth
        
    #     self.logger.info(f"================================================")
    #     self.logger.info(f"depths.shape: {depths.shape}")
    #     self.logger.info(f"================================================\n")
        
    #     # Project 3D points to 2D image plane using camera intrinsics
    #     proj_points = (K @ points.T).T  # [N, 3]
        
    #     self.logger.info(f"================================================")
    #     self.logger.info(f"proj_points.shape: {proj_points.shape}")
    #     self.logger.info(f"================================================\n")

    #     # Perspective division
    #     x_proj = (proj_points[:, 0] / proj_points[:, 2]).astype(int)
    #     y_proj = (proj_points[:, 1] / proj_points[:, 2]).astype(int)
        
    #     # Define image bounds
    #     x_min, x_max = x_proj.min() - self.padding, x_proj.max() + self.padding
    #     y_min, y_max = y_proj.min() - self.padding, y_proj.max() + self.padding
        
    #     self.logger.info(f"================================================")
    #     self.logger.info(f"x_min: {x_min} x_max: {x_max}")
    #     self.logger.info(f"y_min: {y_min} y_max: {y_max}")
    #     self.logger.info(f"================================================\n")

    #     # Shift coordinates to be non-negative
    #     x_proj -= x_min
    #     y_proj -= y_min
        
        
    #     # Create depth buffer (initialize with infinity)
    #     width = x_max - x_min + 1
    #     height = y_max - y_min + 1
    #     depth_buffer = np.full((height, width), np.inf)
        
    #     # Create index buffer to track point indices
    #     index_buffer = np.full((height, width), -1, dtype=int)
        
    #     # Fill depth buffer (keeping track of closest points)
    #     for i, (x, y, d) in enumerate(zip(x_proj, y_proj, depths)):
    #         if 0 <= x < width and 0 <= y < height:
    #             if d < depth_buffer[y, x]:
    #                 depth_buffer[y, x] = d
    #                 index_buffer[y, x] = i
        
    #     # Initialize mask for occluded points
    #     occluded_mask = np.zeros(len(points), dtype=bool)
        
    #     # Mark points as occluded if they're significantly behind the closest point
    #     for i, (x, y, d) in enumerate(zip(x_proj, y_proj, depths)):
    #         if 0 <= x < width and 0 <= y < height:
    #             if d > depth_buffer[y, x] + self.depth_threshold:
    #                 occluded_mask[i] = True
        
    #     # Create color array (default colors preserved)
    #     colors = pcd.point['colors'].numpy()
        
    #     # Set orange color for occluded points [255, 165, 0]
    #     colors[occluded_mask] = np.array([255, 165, 0])
        
    #     # Update colors in output pointcloud
    #     occ_pcd.point['colors'] = o3c.Tensor(colors, dtype=o3c.Dtype.UInt8)
        
    #     self.logger.info(f"=================================")    
    #     self.logger.info(f"Found {occluded_mask.sum()} occluded points")
    #     self.logger.info(f"Total points: {len(points)}")
    #     self.logger.info(f"Percentage occluded: {100 * occluded_mask.sum() / len(points):.2f}%")
    #     self.logger.info(f"=================================\n")
        
    #     return occ_pcd
    
    def apply_occ_mask_to_pcd(self, pcd: o3d.t.geometry.PointCloud, K: np.ndarray = None) -> o3d.t.geometry.PointCloud:
        '''
        Add occlusion mask to the pointcloud.
        '''
        pass
    
    def get_occ_mask(self, pcd: o3d.t.geometry.PointCloud,
                     K: np.ndarray = None,
                     bb: Dict[str, float] = None) -> np.ndarray:
        '''

        Generate occupancy map from the pointcloud using ray-tracing and depth buffer approach.
        Colors occluded points in orange.
        
        Args:
            pcd (o3d.t.geometry.PointCloud): Input pointcloud in camera frame
            K (np.ndarray): 3x3 camera intrinsics matrix
                [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
            
        Returns:
            o3d.t.geometry.PointCloud: Pointcloud with occluded points colored orange
        '''
        assert K.shape == (3, 3), "Camera intrinsics must be a 3x3 matrix"
        
        pcd_cropped = crop_pcd(pcd, bb)

        
        # Create a copy of input pointcloud
        occ_pcd = pcd_cropped.clone()
        
        # Get points as numpy array
        points = pcd_cropped.point['positions'].numpy()
        
        self.logger.info(f"================================================")
        self.logger.info(f"points.shape: {points.shape}")
        self.logger.info(f"================================================\n")

        # Calculate depth (distance from camera plane)
        depths = points[:, 2]  # z-coordinate is depth
        
        self.logger.info(f"================================================")
        self.logger.info(f"depths.shape: {depths.shape}")
        self.logger.info(f"================================================\n")
        
        # Project 3D points to 2D image plane using camera intrinsics
        proj_points = (K @ points.T).T  # [N, 3]
        
        self.logger.info(f"================================================")
        self.logger.info(f"proj_points.shape: {proj_points.shape}")
        self.logger.info(f"================================================\n")

        # Perspective division
        x_proj = (proj_points[:, 0] / proj_points[:, 2]).astype(int)
        y_proj = (proj_points[:, 1] / proj_points[:, 2]).astype(int)
        
        # Define image bounds
        x_min, x_max = x_proj.min() - self.padding, x_proj.max() + self.padding
        y_min, y_max = y_proj.min() - self.padding, y_proj.max() + self.padding
        
        self.logger.info(f"================================================")
        self.logger.info(f"x_min: {x_min} x_max: {x_max}")
        self.logger.info(f"y_min: {y_min} y_max: {y_max}")
        self.logger.info(f"================================================\n")

        # Shift coordinates to be non-negative
        x_proj -= x_min
        y_proj -= y_min
        
        
        # Create depth buffer (initialize with infinity)
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        depth_buffer = np.full((height, width), np.inf)
        
        # Create index buffer to track point indices
        index_buffer = np.full((height, width), -1, dtype=int)
        
        # Fill depth buffer (keeping track of closest points)
        for i, (x, y, d) in enumerate(zip(x_proj, y_proj, depths)):
            if 0 <= x < width and 0 <= y < height:
                if d < depth_buffer[y, x]:
                    depth_buffer[y, x] = d
                    index_buffer[y, x] = i
        
        # Initialize mask for occluded points
        occluded_mask = np.zeros(len(points), dtype=bool)
        
        # Mark points as occluded if they're significantly behind the closest point
        for i, (x, y, d) in enumerate(zip(x_proj, y_proj, depths)):
            if 0 <= x < width and 0 <= y < height:
                if d > depth_buffer[y, x] + self.depth_threshold:
                    occluded_mask[i] = True
        
        # Create color array (default colors preserved)
        colors = pcd.point['colors'].numpy()
        
        # Set orange color for occluded points [255, 165, 0]
        colors[occluded_mask] = np.array([255, 165, 0])
        
        # Update colors in output pointcloud
        occ_pcd.point['colors'] = o3c.Tensor(colors, dtype=o3c.Dtype.UInt8)
        
        self.logger.info(f"=================================")    
        self.logger.info(f"Found {occluded_mask.sum()} occluded points")
        self.logger.info(f"Total points: {len(points)}")
        self.logger.info(f"Percentage occluded: {100 * occluded_mask.sum() / len(points):.2f}%")
        self.logger.info(f"=================================\n")
        
        return occ_pcd
    

