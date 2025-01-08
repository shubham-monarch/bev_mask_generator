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
import matplotlib.pyplot as plt

from scripts.logger import get_logger
from scripts.helpers import crop_pcd

class OcclusionMap:
    """Class for generating occlusion maps from point clouds using depth buffer approach."""

    # Class-level logger
    logger = get_logger("occlusion_map", level=logging.INFO)
    DEPTH_THRESHOLD = 0.1
    Z_MIN = 0.02

    # # Default parameters
    # PADDING: ClassVar[int] = 100  # pixels
    # DEPTH_THRESHOLD: ClassVar[float] = 0.1  # 10cm threshold for occlusion detection

    # # Default parameters
    def __init__(self):
        """Initialize OcclusionMap."""

        OcclusionMap.logger.info(f"===========================")
        OcclusionMap.logger.info(f"OcclusionMap initialized")
        OcclusionMap.logger.info(f"===========================\n")

    @staticmethod
    def get_occ_mask(pcd: o3d.t.geometry.PointCloud,
                     camera_matrix: np.ndarray,
                     bb: Dict[str, float] = None,
                     camera_projection_matrix: np.ndarray = None) -> o3d.t.geometry.PointCloud:
        
        if camera_matrix.shape != (3, 3):
            raise ValueError("Camera intrinsics must be a 3x3 matrix")
        
        # OcclusionMap.logger.info(f"================================================")
        # OcclusionMap.logger.info(f"pcd.point['positions'].shape: {pcd.point['positions'].shape}")
        # OcclusionMap.logger.info(f"================================================\n")

        # Crop point cloud if bounding box provided
        # pcd_cropped = crop_pcd(pcd, bb)
        pcd_cropped = pcd
        
        # OcclusionMap.logger.info(f"================================================")
        # OcclusionMap.logger.info(f"pcd_cropped.point['positions'].shape: {pcd_cropped.point['positions'].shape}")
        # OcclusionMap.logger.info(f"================================================\n")
        
        # OcclusionMap.logger.info(f"================================================")
        # OcclusionMap.logger.info(f"camera_matrix: {camera_matrix}")
        # OcclusionMap.logger.info(f"================================================\n")

        # OcclusionMap.logger.info(f"================================================")
        # OcclusionMap.logger.info(f"camera_projection_matrix: {camera_projection_matrix}")
        # OcclusionMap.logger.info(f"================================================\n")

        # Create working copy
        # occ_pcd = pcd_cropped.clone()
        points = pcd_cropped.point['positions'].numpy()
        
        # Calculate depths and project points
        proj_points = (camera_matrix @ points.T).T  # [N, 3]

        # Apply camera rotation matrix
        camera_rotation_matrix = camera_projection_matrix[:3, :3]
        proj_points = (camera_rotation_matrix @ points.T).T  # [N, 3]
        
        # OcclusionMap.logger.info(f"================================================")
        # OcclusionMap.logger.info(f"Projected points shape: {proj_points.shape}")
        # OcclusionMap.logger.info(f"================================================\n")
        
        
        # Extract x, y, z coordinates from projected points
        x_coords = proj_points[:, 0]
        y_coords = proj_points[:, 1]
        z_coords = points[:, 2]  # Original z-coordinates from the point cloud
        
        # Create a histogram for z-values
        # plt.figure(figsize=(10, 6))
        # plt.hist(z_coords, bins=30, color='blue', alpha=0.7)
        # plt.title('Histogram of Z Coordinates')
        # plt.xlabel('Z Coordinate Value')
        # plt.ylabel('Frequency')
        # plt.grid(axis='y', alpha=0.75)
        # plt.show()
        
        # OcclusionMap.logger.warning(f"================================================")
        # OcclusionMap.logger.warning(f"range of z_coords: ({z_coords.min()}, {z_coords.max()})")
        # OcclusionMap.logger.warning(f"================================================\n")

        # Perspective division
        x_proj = (proj_points[:, 0] / proj_points[:, 2]).astype(int)
        y_proj = (proj_points[:, 1] / proj_points[:, 2]).astype(int)
        
        # OcclusionMap.logger.info(f"================================================")
        # OcclusionMap.logger.info("x_proj shape: %s", x_proj.shape)
        # OcclusionMap.logger.info("y_proj shape: %s", y_proj.shape)
        # OcclusionMap.logger.info(f"================================================\n")

        # # Visualize points after perspective division
        # fig2, ax2 = plt.subplots(figsize=(10, 8))
        # scatter2 = ax2.scatter(x_proj, y_proj, c=z_coords, cmap='viridis', label='Points After Perspective Division')  # Using z_coords instead of depths
        # ax2.set_xlabel('X Projected')
        # ax2.set_ylabel('Y Projected')
        # ax2.set_title('Points After Perspective Division')
        # ax2.legend()
        # plt.colorbar(scatter2, label='Z Coordinate')  # Updated label to reflect z_coords
        # plt.show()

       
        h, w = 1080, 1920
        depth_buffer = np.full((h, w), np.inf)
        EPS = 1e-6
        # Create a mask for valid projected points within the bounds
        valid_mask = (0 <= x_proj) & (x_proj < h) & (0 <= y_proj) & (y_proj < w)
        
        # OcclusionMap.logger.warning(f"================================================")
        # OcclusionMap.logger.warning(f"valid_mask shape: {valid_mask.shape}")
        # OcclusionMap.logger.warning(f"% valid points: {valid_mask.sum() / valid_mask.size * 100:.2f}%")
        # OcclusionMap.logger.warning(f"================================================\n")



        for i, (x, y, d) in enumerate(zip(x_proj[valid_mask], y_proj[valid_mask], z_coords[valid_mask])):
            depth_buffer[x, y] = min(depth_buffer[x, y], d)
        
        # Detect occlusions
        occluded_mask = np.zeros(len(points), dtype=bool)
        for i, (x, y, d) in enumerate(zip(x_proj, y_proj, z_coords)):
            if 0 <= x < h and 0 <= y < w:
                if d > depth_buffer[x, y] + OcclusionMap.DEPTH_THRESHOLD:
                    occluded_mask[i] = True
                    # OcclusionMap.logger.error(f"========================")
                    # OcclusionMap.logger.error(f"occluded point: {i}")
                    # OcclusionMap.logger.error(f"========================")
        
        # OcclusionMap.logger.info(f"================================================")
        # OcclusionMap.logger.info(f"Finished generating occlusion mask!")
        # OcclusionMap.logger.info(f"occluded_mask shape: {occluded_mask.shape}")
        # OcclusionMap.logger.info(f"================================================\n")
            
        # Color occluded points
        colors = pcd_cropped.point['colors'].numpy()
        colors[occluded_mask] = np.array([255, 165, 0])  # Orange

        occ_pcd = pcd_cropped.clone()
        occ_pcd.point['colors'] = o3c.Tensor(colors, dtype=o3c.Dtype.UInt8)
        
        # Log results
        occluded_count = occluded_mask.sum()
        total_points = len(pcd_cropped.point['positions'].numpy())
        occluded_percentage = 100 * occluded_count / total_points
        
        OcclusionMap.logger.info("Occlusion detection complete:")
        OcclusionMap.logger.info("- Found %d occluded points", occluded_count)
        OcclusionMap.logger.info("- Total points: %d", total_points)
        OcclusionMap.logger.info("- Percentage occluded: %.2f%%", occluded_percentage)
        
        return occ_pcd
