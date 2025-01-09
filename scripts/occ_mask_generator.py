'''
Occlusion map generation
'''
#! /usr/bin/env python3

import open3d as o3d
import numpy as np
import open3d.core as o3c
import logging
import cv2
from typing import Dict, Tuple, Union, List
import matplotlib.pyplot as plt

from scripts.logger import get_logger
from scripts.helpers import crop_pcd

class OccMap:
    """Class for generating occlusion maps from point clouds using depth buffer approach."""

    # class-level variables
    logger = get_logger("occlusion_map", level=logging.INFO)
    DEPTH_THRESHOLD = 0.06
    
    @staticmethod
    def generate_pixel_coordinates(pcd: o3d.t.geometry.PointCloud, 
                              K: np.ndarray, 
                              P: np.ndarray) -> np.ndarray:
        """
        Generate pixel coordinates from point cloud using camera intrinsic and extrinsic matrices.
        
        Args:
            pcd: Input point cloud
            K: 3x3 camera intrinsic matrix
            P: 3x4 camera extrinsic matrix [R|t]
            
        Returns:
            np.ndarray: Nx2 array of pixel coordinates
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
                              img_shape: Tuple[int, int] = (1080, 1920)) -> np.ndarray:
        """Generate depth buffer from projected point cloud coordinates.
        
        Args:
            pcd: Input point cloud
            img_coords: Nx2 array of projected 2D coordinates
            img_shape: Tuple of (height, width) for output buffer
            
        Returns:
            np.ndarray: HxW depth buffer containing minimum depth at each pixel
        """
       
        assert pcd.point['positions'].numpy().shape[0] == img_coords.shape[0], \
            "pcd and img_coords must have the same number of points"
        assert img_coords.shape[1] == 2, \
            "img_coords must be a 2D array with shape [N, 2]"

        h, w = img_shape
        
        u, v = img_coords[:, 0], img_coords[:, 1]
        depths: np.ndarray = pcd.point['positions'].numpy()[:, 2]

        assert v.shape == u.shape == depths.shape, \
            "v, u, and depths must have the same shape"

        # Create a mask for valid projected points within the bounds
        valid_mask: np.ndarray = ((0 <= v) & (v < h) & (0 <= u) & (u < w))

        # log results
        OccMap.logger.info(f"===========================")
        OccMap.logger.info(f"Depth buffer generation:")
        OccMap.logger.info(f"- projected points: {len(img_coords)}")
        OccMap.logger.info(f"- valid projected points: {valid_mask.sum()}")
        OccMap.logger.info(f"- valid projected ratio: {valid_mask.sum() / len(img_coords):.2f}")
        OccMap.logger.info(f"===========================\n")

        valid_v: np.ndarray = v[valid_mask]
        valid_u: np.ndarray = u[valid_mask]
        valid_depths: np.ndarray = depths[valid_mask]

        depth_buffer: np.ndarray = np.full((h, w), np.inf)
        
        for u, v, d in zip(valid_u, valid_v, valid_depths):
            depth_buffer[v, u] = min(depth_buffer[v, u], d)
        
        return depth_buffer


    @staticmethod
    def visualize_projected_points(pcd: o3d.t.geometry.PointCloud,
                                 img_coords: np.ndarray,
                                 img_shape: Tuple[int, int] = (1080, 1920),
                                 output_shape: Tuple[int, int] = (640, 720),
                                 window_name: str = "Projected Points",
                                 point_size: int = 3,
                                 wait_key: bool = True) -> None:
        """Visualize projected points from point cloud onto a 2D image plane.

        Args:
            pcd: Input point cloud containing positions and colors
            img_coords: Nx2 array of projected 2D coordinates
            img_shape: Tuple of (height, width) for output image
            output_shape: Tuple of (height, width) for output image
            window_name: Name of the visualization window
            point_size: Radius of visualized points
            wait_key: Whether to wait for key press before closing window
        """
        try:
            colors: np.ndarray = pcd.point['colors'].numpy()
            h, w = img_shape
            img: np.ndarray = np.zeros((h, w, 3), dtype=np.uint8)
            
            # create mask for valid points
            valid_mask: np.ndarray = ((0 <= img_coords[:, 0]) & (img_coords[:, 0] < w) & 
                                    (0 <= img_coords[:, 1]) & (img_coords[:, 1] < h))
            
            # only process valid points
            valid_coords: np.ndarray = img_coords[valid_mask]
            valid_colors: np.ndarray = colors[valid_mask]
            
            for point, color in zip(valid_coords, valid_colors):
                x, y = int(point[0]), int(point[1])
                # convert rgb to bgr for opencv
                color_bgr: Tuple[int, int, int] = (int(color[2]), int(color[1]), int(color[0]))
                cv2.circle(img, (x, y), point_size, color_bgr, -1)

            # resize image to output shape
            img = cv2.resize(img, output_shape)

            cv2.imshow(window_name, img)
            if wait_key:
                cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # log results
            OccMap.logger.info(f"===========================")
            OccMap.logger.info(f"Projected points visualization:")
            OccMap.logger.info(f"- total points: {len(img_coords)}")
            OccMap.logger.info(f"- valid points: {valid_mask.sum()}")
            OccMap.logger.info(f"- valid ratio: {valid_mask.sum() / len(img_coords):.2f}")
            OccMap.logger.info(f"===========================\n")
            
        except Exception as e:
            OccMap.logger.error(f"===========================")
            OccMap.logger.error(f"Error visualizing projected points: {str(e)}")
            OccMap.logger.error(f"===========================\n")
            
    @staticmethod
    def add_mask_color_to_pcd(pcd: o3d.t.geometry.PointCloud,
                             masks: Union[np.ndarray, List[np.ndarray]],
                             colors: Union[np.ndarray, List[np.ndarray]]) -> o3d.t.geometry.PointCloud:
        """Color points in point cloud based on multiple masks.

        Args:
            pcd: Input point cloud
            masks: Single boolean mask or list of boolean masks indicating points to color
            colors: Single RGB color array or list of RGB colors corresponding to masks
                   (default: orange [255, 165, 0])
        Returns:
            o3d.t.geometry.PointCloud: Point cloud with mask points colored
        """
        # convert single mask/color to list for consistent processing
        if not isinstance(masks, list):
            masks = [masks]
            colors = [colors]
            
        if len(masks) != len(colors):
            raise ValueError(f"number of masks ({len(masks)}) must match number of colors ({len(colors)})")
            
        # color points
        pcd_colors: np.ndarray = pcd.point['colors'].numpy()
        total_masked_points = 0
        
        # apply each mask with its corresponding color
        for mask, color in zip(masks, colors):
            if mask.shape[0] != len(pcd_colors):
                raise ValueError(f"mask length ({mask.shape[0]}) must match number of points ({len(pcd_colors)})")
            pcd_colors[mask] = color
            total_masked_points += mask.sum()

        pcd.point['colors'] = o3c.Tensor(pcd_colors, dtype=o3c.Dtype.UInt8)
        
        # log results
        # OccMap.logger.info(f"===========================")
        # OccMap.logger.info(f"Mask coloring complete:")
        # OccMap.logger.info(f"- total points: {len(pcd.point['positions'].numpy())}")
        # OccMap.logger.info(f"- mask points: {total_masked_points}")
        # OccMap.logger.info(f"- % mask: {100 * total_masked_points / len(pcd.point['positions'].numpy()):.2f}%")
        # OccMap.logger.info(f"===========================\n")
        
        return pcd
    

    @staticmethod
    def get_occ_pcd(pcd: o3d.t.geometry.PointCloud,
                     K: np.ndarray,
                     P: np.ndarray,
                     bb: Dict[str, float] = None,
                     to_crop: bool = True,
                     img_shape: Tuple[int, int] = (1080, 1920)) -> o3d.t.geometry.PointCloud:
        """Generate point cloud with occluded points colored.
        
        Args:
            pcd: Input point cloud
            K: 3x3 camera intrinsic matrix
            P: 3x4 camera extrinsic matrix [R|t]
            bb: Optional bounding box for cropping
            to_crop: Whether to crop the point cloud
            img_shape: Tuple of (height, width) for projection
            
        Returns:
            o3d.t.geometry.PointCloud: Point cloud with occluded points colored
        """
        assert K.shape == (3, 3), "camera_intrinsic_matrix must be a 3x3 matrix"
        assert P.shape == (3, 4), "camera_extrinsic_matrix must be a 3x4 matrix"

        OccMap.logger.info(f"===========================")
        OccMap.logger.info(f"P: \n{P}")
        OccMap.logger.info(f"===========================\n")

        # crop pcd
        pcd_cropped = crop_pcd(pcd, bb) if to_crop else pcd
        
        # [N, 2]
        img_coords: np.ndarray = OccMap.generate_pixel_coordinates(pcd_cropped, K, P)
        
        # optionally visualize the projected points
        OccMap.visualize_projected_points(pcd_cropped, img_coords, img_shape)

        # [h, w]
        depth_buffer: np.ndarray = OccMap.generate_depth_buffer(pcd_cropped, img_coords)

        h, w = img_shape
        u, v = img_coords[:, 0], img_coords[:, 1]
        depths = pcd_cropped.point['positions'].numpy()[:, 2]
        
        # detect occlusions
        hidden_mask: np.ndarray = np.zeros(len(depths), dtype=bool)
        bound_mask: np.ndarray = np.zeros(len(depths), dtype=bool)
        
        for i, (u, v, d) in enumerate(zip(u, v, depths)):
            if 0 <= u < w and 0 <= v < h:
                if d > depth_buffer[v, u] + OccMap.DEPTH_THRESHOLD:
                    hidden_mask[i] = True
            else:
                # occ_mask[i] = True
                bound_mask[i] = True
            
        # color points using the new method
        hidden_color = np.array([241, 196, 15])     # sun yellow
        bound_color = np.array([142, 68, 173])   # wisteria purple
        
        OccMap.logger.info(f"===========================")
        OccMap.logger.info(f"Occ-mask generation:")
        OccMap.logger.info(f"- total points: {len(depths)}")
        OccMap.logger.info(f"- hidden points: {hidden_mask.sum()}")
        OccMap.logger.info(f"- bound points: {bound_mask.sum()}")
        OccMap.logger.info(f"- % hidden: {100 * hidden_mask.sum() / len(depths):.2f}%")
        OccMap.logger.info(f"- % bound: {100 * bound_mask.sum() / len(depths):.2f}%")
        OccMap.logger.info(f"- % occ: {100 * (hidden_mask.sum() + bound_mask.sum()) / len(depths):.2f}%")
        OccMap.logger.info(f"===========================\n")

        OccMap.add_mask_color_to_pcd(pcd_cropped, [hidden_mask, bound_mask], [hidden_color, bound_color])
        return pcd_cropped
