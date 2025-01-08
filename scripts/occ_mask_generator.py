'''
Occlusion map generation
'''
#! /usr/bin/env python3

import open3d as o3d
import numpy as np
import open3d.core as o3c
import logging
import cv2
from typing import Dict, Tuple
import matplotlib.pyplot as plt

from scripts.logger import get_logger
from scripts.helpers import crop_pcd

class OccMap:
    """Class for generating occlusion maps from point clouds using depth buffer approach."""

    # class-level variables
    logger = get_logger("occlusion_map", level=logging.INFO)
    
    DEPTH_THRESHOLD = 0.1
    Z_MIN = 0.02

    def __init__(self):
        """Initialize OcclusionMap."""

        OccMap.logger.info(f"===========================")
        OccMap.logger.info(f"OccMap initialized")
        OccMap.logger.info(f"===========================\n")

    @staticmethod
    def generate_pixel_coordinates(pcd: o3d.t.geometry.PointCloud, 
                              K: np.ndarray, 
                              P: np.ndarray) -> np.ndarray:
        """
        :param pcd -> [N, 3] point cloud
        :param K -> [3 * 3] camera intrinsic matrix
        :param P -> [3 * 4] camera extrinsic matrix --> [R | t]
        """
        
        assert K.shape == (3, 3), "camera_intrinsic_matrix must be a 3x3 matrix"
        assert P.shape == (3, 4), "camera_extrinsic_matrix must be a 3x4 matrix"

        X: np.ndarray = pcd.point['positions'].numpy() # [N, 3]
        X_homo: np.ndarray = np.hstack([X, np.ones((X.shape[0], 1))]) # [N, 4]

        # camera projection matrix -->  K * [R | t] 
        M: np.ndarray = K @ P # [3 * 4]

        # project points to 2D image plane
        x_homo: np.ndarray = (M @ X_homo.T).T # [N, 3]

        # perspective division
        x_homo: np.ndarray = x_homo / x_homo[:, 2:3] # [N, 3]
        
        # non-homogeneous coordinates
        x: np.ndarray = np.floor(x_homo[:, :2]).astype(int) # [N, 2]

        assert x.shape == (X.shape[0], 2), "pixel coordinates must be a 2D array"
        return x
    
    @staticmethod
    def generate_depth_buffer(pcd: o3d.t.geometry.PointCloud,
                              img_coords: np.ndarray,
                              EPS: float = 1e-6,
                              img_shape: Tuple[int, int] = (1080, 1920)) -> np.ndarray:
        
        assert pcd.point['positions'].numpy().shape[0] == img_coords.shape[0], \
            "pcd and img_coords must have the same number of points"
        assert img_coords.shape[1] == 2, \
            "img_coords must be a 2D array with shape [N, 2]"

        h, w = img_shape
        
        u, v = img_coords[:, 0], img_coords[:, 1]
        depths = pcd.point['positions'].numpy()[:, 2]

        assert u.shape == v.shape == depths.shape, \
            "u, v, and depths must have the same shape"

        # Create a mask for valid projected points within the bounds
        valid_mask: np.ndarray = ((0 <= u) & (u < h) & (0 <= v) & (v < w))
        
        OccMap.logger.warning(f"===========================")
        OccMap.logger.warning(f"valid_ratio: {valid_mask.sum() / len(valid_mask)}")
        OccMap.logger.warning(f"===========================\n")

        valid_u: np.ndarray = u[valid_mask]
        valid_v: np.ndarray = v[valid_mask]
        valid_depths: np.ndarray = depths[valid_mask]

        depth_buffer: np.ndarray = np.full((h, w), np.inf)
        
        for u, v, d in zip(valid_u, valid_v, valid_depths):
            depth_buffer[u, v] = min(depth_buffer[u, v], d)
        
        return depth_buffer

    @staticmethod
    def get_occ_mask(pcd: o3d.t.geometry.PointCloud,
                     K: np.ndarray,
                     P: np.ndarray,
                     bb: Dict[str, float] = None,
                     img_shape: Tuple[int, int] = (1080, 1920)) -> o3d.t.geometry.PointCloud:
        
        """
        :param pcd -> [N, 3] point cloud
        :param K -> [3 * 3] camera intrinsic matrix
        :param P -> [3 * 4] camera extrinsic matrix --> [R | t]
        """

        assert K.shape == (3, 3), "camera_intrinsic_matrix must be a 3x3 matrix"
        assert P.shape == (3, 4), "camera_extrinsic_matrix must be a 3x4 matrix"
        
        # crop pcd
        pcd_cropped = crop_pcd(pcd, bb)

        # [N, 2]
        img_coords: np.ndarray = OccMap.generate_pixel_coordinates(pcd_cropped, K, P)

        import matplotlib.pyplot as plt
        
        # Visualize the image coordinates
        plt.figure(figsize=(10, 10))
        plt.scatter(img_coords[:, 0], img_coords[:, 1], c='blue', marker='o', label='Projected Points')
        plt.xlim(0, img_shape[1])
        plt.ylim(img_shape[0], 0)  # Invert y-axis to match image coordinates
        plt.title('Visualization of Projected Image Coordinates')
        plt.xlabel('Image Width (u)')
        plt.ylabel('Image Height (v)')
        plt.legend()
        plt.grid()
        plt.show()

        # [h, w]
        depth_buffer: np.ndarray = OccMap.generate_depth_buffer(pcd_cropped, img_coords)

        h, w = img_shape
        u, v = img_coords[:, 0], img_coords[:, 1]
        depths = pcd_cropped.point['positions'].numpy()[:, 2]
        
        # detect occlusions
        occ_mask: np.ndarray = np.zeros(len(depths), dtype=bool)
        for i, (u, v, d) in enumerate(zip(u, v, depths)):
            if 0 <= u < h and 0 <= v < w:
                if d > depth_buffer[u, v] + OccMap.DEPTH_THRESHOLD:
                    occ_mask[i] = True
                
            
        # color occluded points
        colors: np.ndarray = pcd_cropped.point['colors'].numpy()
        colors[occ_mask] = np.array([255, 165, 0])  # Orange

        occ_pcd: o3d.t.geometry.PointCloud = pcd_cropped.clone()
        occ_pcd.point['colors'] = o3c.Tensor(colors, dtype=o3c.Dtype.UInt8)
        
        # log results
        occ_count: int = occ_mask.sum()
        total_points: int = len(pcd_cropped.point['positions'].numpy())
        occ_percentage: float = 100 * occ_count / total_points

        OccMap.logger.info(f"===========================")
        OccMap.logger.info(f"Occlusion detection complete:")
        OccMap.logger.info(f"- % occluded: {occ_percentage:.2f}%")
        OccMap.logger.info(f"===========================\n")
        
        return occ_pcd