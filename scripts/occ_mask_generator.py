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
import traceback

from scripts.logger import get_logger
from scripts.helpers import crop_pcd



class OccMap:
    """Class for generating occlusion maps from point clouds using depth buffer approach."""

    # class-level variables
    logger = get_logger("occlusion_map", level=logging.INFO)
    # DEPTH_THRESHOLD = 0.06
    DEPTH_THRESHOLD = 0.1
    
    @staticmethod
    def get_pixel_coordinates(pcd: o3d.t.geometry.PointCloud, 
                              K: np.ndarray, 
                              P: np.ndarray = None) -> np.ndarray:
        """
        Generate pixel coordinates from point cloud using camera intrinsic and extrinsic matrices.
        
        Args:
            pcd: Input point cloud
            K: 3x3 camera intrinsic matrix
            P: 3x4 camera extrinsic matrix [R|t]
            
        Returns:
            np.ndarray: Nx2 array of pixel coordinates
        """
        
        assert K is not None and K.shape == (3, 3), "camera_intrinsic_matrix must be a 3x3 matrix"
        assert P is not None and P.shape == (3, 4), "camera_extrinsic_matrix must be a 3x4 matrix"

        # if P is None:
        #     # create a 3x4 identity transformation matrix [R|t]
        #     # where R is 3x3 identity and t is zero translation
        #     P = np.hstack([np.eye(3), np.zeros((3,1))])

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
    def generate_depth_buffer(sfm_pcd: o3d.t.geometry.PointCloud = None,
                              stereo_pcd: o3d.t.geometry.PointCloud = None,
                              K: np.ndarray = None,
                              P: np.ndarray = None,
                              depth_buffer_shape: Tuple[int, int] = (1080, 1920),
                              logging_level: int = logging.INFO) -> np.ndarray:
        """Generates a depth buffer by projecting the sfm point cloud onto a 2D image plane.

        This function projects the sfm point cloud onto a 2D image plane using the provided
        camera intrinsics matrix (K). It then creates a depth buffer where each pixel stores
        the minimum depth value of the projected points. This buffer is used to determine
        occlusion by comparing the depth of other points with the values in this buffer.

        Args:
            sfm_pcd: Input sfm point cloud.
            stereo_pcd: Input stereo point cloud (not used in this function, but kept for consistency).
            K: 3x3 camera intrinsic matrix.
            depth_buffer_shape: Tuple of (height, width) for the output depth buffer.
            logging_level: Logging level for the function.

        Returns:
            np.ndarray: A HxW depth buffer containing the minimum depth at each pixel.
        """
        assert sfm_pcd is not None, "sfm_pcd is required"
        assert K is not None and K.shape == (3, 3), "K must be a 3x3 matrix"
        assert P is not None and P.shape == (3, 4), "P must be a 3x4 matrix"
        assert (depth_buffer_shape == (1080, 1920)), "depth_buffer_shape must be (1080, 1920)"
       
        logger = get_logger("generate_depth_buffer", level=logging_level)

        h, w = depth_buffer_shape
       
        # project points to image plane
        sfm_img_projection: np.ndarray = OccMap.get_pixel_coordinates(sfm_pcd, K, P)
        
        # logger.info("───────────────────────────────")
        # logger.info(f"sfm_img_projection.shape: {sfm_img_projection.shape}")
        # logger.info("───────────────────────────────")

        # projections must lie within the depth-mask bounds
        sfm_mask: np.ndarray = ((0 <= sfm_img_projection[:, 1]) & (sfm_img_projection[:, 1] < h) & 
                                  (0 <= sfm_img_projection[:, 0]) & (sfm_img_projection[:, 0] < w))
        
        # logger.info("───────────────────────────────")
        # logger.info(f"sfm_mask.shape: {sfm_mask.shape}")
        # logger.info("───────────────────────────────")
        
        # filter valid projections
        sfm_img_projection: np.ndarray = sfm_img_projection[sfm_mask]

        logger.error("───────────────────────────────")
        logger.error(f"sfm_img_projection.shape: {sfm_img_projection.shape}")
        logger.error("───────────────────────────────")
        
        # filter points with valid depths
        sfm_depths: np.ndarray = sfm_pcd.point['positions'].numpy()[:, 2]

        # logger.info("───────────────────────────────")
        # logger.info(f"sfm_depths.shape: {sfm_depths.shape}")
        # logger.info("───────────────────────────────")

        sfm_depths = sfm_depths[sfm_mask]
        
        # logger.info("───────────────────────────────")
        # logger.info(f"sfm_depths.shape: {sfm_depths.shape}")
        # logger.info("───────────────────────────────")
        
        # generate depth buffer
        depth_buffer: np.ndarray = np.full((h, w), np.inf)

        assert sfm_depths.shape[0] == sfm_img_projection.shape[0], \
            f"The number of points in sfm_depths " \
            f"({sfm_depths.shape}) " \
            f"must match the number of points in sfm_img_projection " \
            f"({sfm_img_projection.shape})"
        
        # update min-depth for each depth-buffer pixel with sfm-depths
        for u, v, d in zip(sfm_img_projection[:, 0], sfm_img_projection[:, 1], sfm_depths):
            if d > 0:
                depth_buffer[v, u] = min(depth_buffer[v, u], d)
            else:
                depth_buffer[v, u] = np.inf
        
        
        if stereo_pcd is not None:
            stereo_img_projection: np.ndarray = OccMap.get_pixel_coordinates(stereo_pcd, K, P)
        
            stereo_mask: np.ndarray = ((0 <= stereo_img_projection[:, 1]) & (stereo_img_projection[:, 1] < h) & 
                                      (0 <= stereo_img_projection[:, 0]) & (stereo_img_projection[:, 0] < w))

            stereo_img_projection: np.ndarray = stereo_img_projection[stereo_mask]

            stereo_depths: np.ndarray = stereo_pcd.point['positions'].numpy()[:, 2]
            stereo_depths = stereo_depths[stereo_mask]

            assert stereo_depths.shape[0] == stereo_img_projection.shape[0], \
                f"The number of points in stereo_depths " \
                f"({stereo_depths.shape}) " \
                f"must match the number of points in stereo_img_projection " \
                f"({stereo_img_projection.shape})"

            # update depth-buffer pixel with stereo-depths    
            for u, v, d in zip(stereo_img_projection[:, 0], stereo_img_projection[:, 1], stereo_depths):
                if d > 0:
                    depth_buffer[v, u] = min(depth_buffer[v, u], d)
                else:
                    depth_buffer[v, u] = np.inf

        return depth_buffer
    
    @staticmethod
    def project_pcd_to_img(pcd: o3d.t.geometry.PointCloud,
                           K: np.ndarray,
                           P: np.ndarray = None,
                           img_shape: Tuple[int, int] = (1080, 1920),
                           visualize: bool = True) -> np.ndarray:
        """Project point cloud to image plane and visualize.

        Args:
            pcd: Input point cloud
            K: 3x3 camera intrinsic matrix
            P: 3x4 camera extrinsic matrix [R|t]
            img_shape: Tuple of (height, width) for output image
            
        Returns:
            np.ndarray: Visualization image showing projected points
        """

        assert pcd is not None, "pcd is required"
        assert K is not None, "K is required"
        assert img_shape == (1080, 1920), "img_shape must be (1080, 1920)"

        img_coords: np.ndarray = OccMap.get_pixel_coordinates(pcd, K, P)

        # visualize projected points
        img: np.ndarray = OccMap.visualize_projected_points(pcd, 
                                                            img_coords, 
                                                            img_shape, 
                                                            output_shape = (1080, 1920),
                                                            visualize=visualize)

        return img
        
    @staticmethod
    def visualize_projected_points(pcd: o3d.t.geometry.PointCloud,
                                 img_coords: np.ndarray,
                                 img_shape: Tuple[int, int] = (1080, 1920),
                                 output_shape: Tuple[int, int] = (640, 720),
                                 window_name: str = "Projected Points",
                                 point_size: int = 3,
                                 visualize: bool = True
                                 ) -> np.ndarray:
        """Visualize projected points from point cloud onto a 2D image plane.

        Args:
            pcd: Input point cloud containing positions and colors
            img_coords: Nx2 array of projected 2D coordinates
            img_shape: Tuple of (height, width) for output image
            output_shape: Tuple of (height, width) for output image
            window_name: Name of the visualization window
            point_size: Radius of visualized points
            visualize: Whether to visualize the image
        Returns:
            np.ndarray: Visualization image showing projected points
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

            if visualize:
                # resize image to output shape
                # img = cv2.resize(img, output_shape)
                cv2.imshow(window_name, img)
                
                # wait for 'q' key to be pressed
                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        cv2.destroyAllWindows()
                        break
            
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

        return img
            
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
    def get_sfm_occ_pcd(pcd: o3d.t.geometry.PointCloud,
                     K: np.ndarray,
                     P: np.ndarray = None,
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
        # assert P.shape == (3, 4), "camera_extrinsic_matrix must be a 3x4 matrix"


        logger = get_logger("get_occ_pcd", level=logging.INFO)
        if P is None:
            # create a 3x4 identity transformation matrix [R|t]
            # where R is 3x3 identity and t is zero translation
            P = np.hstack([np.eye(3), np.zeros((3,1))])

        # crop pcd
        pcd_cropped = crop_pcd(pcd, bb) if to_crop else pcd
        
        # [N, 2]
        img_coords: np.ndarray = OccMap.get_pixel_coordinates(pcd_cropped, K, P)

        
        # optionally visualize the projected points
        OccMap.visualize_projected_points(pcd_cropped, 
                                          img_coords, 
                                          img_shape, 
                                          output_shape = (1080, 1920),
                                          visualize=False)
        
        # [h, w]
        depth_buffer: np.ndarray = OccMap.generate_depth_buffer(sfm_pcd = pcd_cropped,
                                                                 K = K)

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
        OccMap.logger.info(f"SFM OCC-MASK GENERATION:")
        OccMap.logger.info(f"- total points: {len(depths)}")
        OccMap.logger.info(f"- hidden points: {hidden_mask.sum()}")
        OccMap.logger.info(f"- bound points: {bound_mask.sum()}")
        OccMap.logger.info(f"- % hidden: {100 * hidden_mask.sum() / len(depths):.2f}%")
        OccMap.logger.info(f"- % bound: {100 * bound_mask.sum() / len(depths):.2f}%")
        OccMap.logger.info(f"- % occ: {100 * (hidden_mask.sum() + bound_mask.sum()) / len(depths):.2f}%")
        OccMap.logger.info(f"===========================\n")

        OccMap.add_mask_color_to_pcd(pcd_cropped, [hidden_mask, bound_mask], [hidden_color, bound_color])
        return pcd_cropped

    @staticmethod
    def visualize_matches(left_img: np.ndarray,
                     right_img: np.ndarray,
                     kp1: List[cv2.KeyPoint],
                     kp2: List[cv2.KeyPoint],
                     good_matches: List[cv2.DMatch],
                     window_name: str = "Stereo Matches",
                     wait_key: bool = True) -> np.ndarray:
        """Visualize matching keypoints between stereo image pair.
        
        Args:
            left_img: Left stereo image
            right_img: Right stereo image
            kp1: Keypoints detected in left image
            kp2: Keypoints detected in right image
            good_matches: List of good matches between keypoints
            window_name: Name of display window
            wait_key: Whether to wait for key press
            
        Returns:
            np.ndarray: Visualization image showing matches
        """
        try:
            good_matches = good_matches[:50]
            # create match visualization image
            match_img = cv2.drawMatches(left_img, kp1,
                                      right_img, kp2,
                                      good_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            # draw horizontal lines to help verify rectification
            h, w = left_img.shape[:2]
            line_interval = h // 50
            for y in range(0, h, line_interval):
                cv2.line(match_img, (0, y), (w*2, y), (0, 255, 0), 1)
                
            # display image
            cv2.imshow(window_name, match_img)
            if wait_key:
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            return match_img
            
        except Exception as e:
            OccMap.logger.error(f"===========================")
            OccMap.logger.error(f"Error visualizing matches: {str(e)}")
            OccMap.logger.error(f"===========================\n")
            raise

    @staticmethod
    def is_rectified(left_img: np.ndarray, 
                     right_img: np.ndarray,
                     num_keypoints: int = 1000,
                     epipolar_threshold: float = 1.0,
                     visualize: bool = True) -> Tuple[bool, float]:
        """Check if a stereo image pair is rectified by analyzing epipolar geometry.
        
        In rectified images, corresponding points should lie on the same horizontal scanline.
        This method:
        1. Detects keypoints using ORB detector in both images
        2. Finds matches between keypoints using Brute Force matcher
        3. Calculates the vertical disparity between matched points
        4. If average vertical disparity is below threshold, considers images rectified
        
        Args:
            left_img: Left stereo image
            right_img: Right stereo image
            num_keypoints: Number of keypoints to detect (default: 1000)
            epipolar_threshold: Maximum allowed average vertical disparity in pixels (default: 1.0)
            visualize: Whether to visualize the matches (default: True)
            
        Returns:
            Tuple[bool, float]: (is_rectified, average_vertical_disparity)
        """
        try:
            # convert images to grayscale if needed
            if len(left_img.shape) == 3:
                left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            else:
                left_gray = left_img.copy()
                right_gray = right_img.copy()
                # convert back to BGR for visualization
                left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
                right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
                
            # verify images have same dimensions
            if left_gray.shape != right_gray.shape:
                raise ValueError("input images must have same dimensions")
                
            # initialize ORB detector
            orb = cv2.ORB_create(nfeatures=num_keypoints)
            
            # detect keypoints and compute descriptors
            kp1, des1 = orb.detectAndCompute(left_gray, None)
            kp2, des2 = orb.detectAndCompute(right_gray, None)
            
            if len(kp1) < 10 or len(kp2) < 10:
                raise ValueError("not enough keypoints detected in images")
                
            # create BFMatcher and match descriptors
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            # sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) < 10:
                raise ValueError("not enough good matches found between images")
                
            # calculate vertical disparities
            vertical_disparities = []
            good_matches = []
            
            for match in matches:
                # get coordinates of matched keypoints
                pt1 = kp1[match.queryIdx].pt
                pt2 = kp2[match.trainIdx].pt
                
                # calculate vertical disparity
                vertical_disparity = abs(pt1[1] - pt2[1])
                
                # only keep matches with reasonable vertical disparity
                if vertical_disparity < 50:  # filter out obvious outliers
                    vertical_disparities.append(vertical_disparity)
                    good_matches.append(match)
            
            if len(vertical_disparities) < 10:
                raise ValueError("not enough valid matches after filtering")
                
            avg_vertical_disparity = np.mean(vertical_disparities)
            is_rectified = avg_vertical_disparity < epipolar_threshold
            
            # visualize matches if requested
            if visualize:
                OccMap.visualize_matches(left_img, right_img, kp1, kp2, good_matches[:50])
            
            # log results
            OccMap.logger.info(f"===========================")
            OccMap.logger.info(f"Rectification check:")
            OccMap.logger.info(f"- number of keypoints: {len(kp1)}/{len(kp2)}")
            OccMap.logger.info(f"- number of good matches: {len(good_matches)}")
            OccMap.logger.info(f"- average vertical disparity: {avg_vertical_disparity:.2f} pixels")
            OccMap.logger.info(f"- is rectified: {is_rectified}")
            OccMap.logger.info(f"===========================\n")
            
            return is_rectified, avg_vertical_disparity
            
        except Exception as e:
            OccMap.logger.error(f"===========================")
            OccMap.logger.error(f"Error checking rectification: {str(e)}")
            OccMap.logger.error(f"===========================\n")
            raise


    @staticmethod
    def get_stereo_disparity(img_L: np.ndarray, img_R: np.ndarray) -> np.ndarray:
        """Compute disparity map for stereo image pair.
        
        Args:
            left_image: Left stereo image
            right_image: Right stereo image
            
        Returns:
            np.ndarray: Disparity map
        """
        
        # img_L: np.ndarray = cv2.resize(left_image, (480, 640))
        # img_R: np.ndarray = cv2.resize(right_image, (480, 640))

        window_size: int = 3
        min_disp: int = -2
        max_disp: int = 160

        stereo_sgbm = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=max_disp - min_disp,
            blockSize=window_size,
            P1=8 * 3 * window_size**2,
            P2=32 * 3 * window_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        disparity: np.ndarray = stereo_sgbm.compute(img_L, img_R).astype(np.float32) / 16.0
        
        # set invalid disparity values to NaN
        valid_disparity_mask: np.ndarray = (disparity >= min_disp) & (disparity <= max_disp)
        disparity[~valid_disparity_mask] = np.nan

        OccMap.logger.info(f"===========================")
        OccMap.logger.info(f"nan values: {np.isnan(disparity).sum()} |"
                           f" {np.isnan(disparity).sum() / disparity.size:.2f}%")
        OccMap.logger.info(f"===========================\n")
        
        return disparity

    @staticmethod       
    def get_stereo_pcd(left_img: np.ndarray, right_img: np.ndarray, 
                       K: np.ndarray,
                       baseline: float,
                       fill_nan_value = None
                       ) -> o3d.t.geometry.PointCloud:
        """Generate point cloud from stereo image pair.
        
        Args:
            left_img: Left stereo image --> (1080, 1920)
            right_img: Right stereo image --> (1080, 1920)
            K: Camera intrinsic matrix --> (3, 3)
            baseline: Baseline distance between stereo cameras in meters --> 0.12
            fill_nan_value: Value to fill NaN disparities with (default: None, which defaults to 0)
            
        Returns:
            o3d.t.geometry.PointCloud: Point cloud with colors
        """

        assert left_img.shape == (1080,1920,3), f"Left image shape must be (1080,1920,3) but is {left_img.shape}"
        assert right_img.shape == (1080,1920,3), f"Right image shape must be (1080,1920,3) but is {right_img.shape}"
        assert fill_nan_value is not None and isinstance(fill_nan_value, int), f"fill_nan_value must be an integer"

        # get disparity map
        disparity: np.ndarray = OccMap.get_stereo_disparity(left_img, right_img)

        assert disparity.shape == left_img.shape[:2], "Disparity map shape does not match left image"
        
        nan_count: int = np.isnan(disparity).sum()
        # OccMap.logger.info(f"===========================")
        # OccMap.logger.info(f"[get_stereo_pcd] Count of NaN values in disparity: {nan_count}")
        # OccMap.logger.info(f"===========================\n")

        # replace nan values with fill_nan_value for depth calculation
        disparity = np.nan_to_num(disparity, nan=fill_nan_value)
        
        # get image dimensions
        h, w = disparity.shape
        
        # create mesh grid for pixel coordinates
        v, u = np.mgrid[0:h, 0:w]
        
        # compute 3d points
        focal_length: float = K[0, 0]
        
        # avoid division by zero in depth calculation
        mask: np.ndarray = disparity > 0
        depth: np.ndarray = np.zeros_like(disparity)
        depth[mask] = (focal_length * baseline) / disparity[mask]
        
        # calculate 3D coordinates
        x: np.ndarray = (u - K[0, 2]) * depth / focal_length
        y: np.ndarray = (v - K[1, 2]) * depth / focal_length
        z: np.ndarray = depth

        # stack coordinates
        points: np.ndarray = np.stack([x, y, z], axis=-1)
        
        # filter out points with zero depth
        valid_points_mask: np.ndarray = depth > 0
        points = points[valid_points_mask]

        
        
        # get colors from resized left image
        colors: np.ndarray = left_img.astype(np.uint8)
        colors = colors[valid_points_mask]

        # create open3d tensor pointcloud
        pcd = o3d.t.geometry.PointCloud()
        pcd.point.positions = o3d.core.Tensor(points.astype(np.float32))
        pcd.point.colors = o3d.core.Tensor(colors.astype(np.uint8))

        OccMap.logger.info(f"===========================")
        OccMap.logger.info(f"[get_stereo_pcd] Generated pointcloud with {len(points)} points")
        OccMap.logger.info(f"===========================\n")
        
        return pcd

    @staticmethod
    def combine_pcds(pcds: List[o3d.t.geometry.PointCloud]) -> o3d.t.geometry.PointCloud:
        """Combine multiple point clouds, preserving original colors.
        
        Args:
            pcds: List of point clouds to combine.
            
        Returns:
            o3d.t.geometry.PointCloud: Combined point cloud with original colors.
        """
        assert isinstance(pcds, list), f"pcds must be a list, but got {type(pcds)}"
        assert len(pcds) > 0, f"pcds must have at least one point cloud, but got {len(pcds)}"
        
        position_tensors: List[np.ndarray] = [pcd.point['positions'].numpy() for pcd in pcds]
        stacked_positions: o3c.Tensor = o3c.Tensor(np.vstack(position_tensors), dtype=o3c.Dtype.Float32)
        
        color_tensors: List[np.ndarray] = [pcd.point['colors'].numpy() for pcd in pcds]
        stacked_colors: o3c.Tensor = o3c.Tensor(np.vstack(color_tensors), dtype=o3c.Dtype.UInt8)

        # create a unified point cloud
        map_to_tensors: Dict[str, o3c.Tensor] = {
            'positions': stacked_positions,
            'colors': stacked_colors        
        }

        combined_pcd: o3d.t.geometry.PointCloud = o3d.t.geometry.PointCloud(map_to_tensors)    
        return combined_pcd

    @staticmethod
    def color_pcd(pcd: o3d.t.geometry.PointCloud,
                  color: np.ndarray = np.array([173, 216, 230, 200]),  # light blue with alpha=200
                  ) -> o3d.t.geometry.PointCloud:
        """Color all points in a point cloud with a specified color.
        
        Args:
            pcd: Input point cloud
            color: RGBA color array (default: semi-transparent light blue)
                   Values should be in range [0, 255] for RGB and alpha
                   If RGB color is provided without alpha, full opacity is assumed
        
        Returns:
            o3d.t.geometry.PointCloud: Point cloud with all points colored
            
        Note:
            Alpha value in color is used for visualization only when rendering
            the point cloud. The stored colors in the point cloud will be RGB.
        """
        # ensure input point cloud is not modified
        colored_pcd = o3d.t.geometry.PointCloud(pcd)
        
        # handle both RGB and RGBA color inputs
        if len(color) == 4:
            rgb_color = color[:3]
            alpha = color[3] / 255.0  # normalize alpha to [0,1]
            # apply alpha by blending with white
            rgb_color = (rgb_color * alpha + 255 * (1 - alpha)).astype(np.uint8)
        else:
            rgb_color = color
        
        # create color array with same color for all points
        num_points = len(colored_pcd.point['positions'])
        colors = np.tile(rgb_color, (num_points, 1))
        
        # assign colors to point cloud
        colored_pcd.point['colors'] = o3c.Tensor(colors, dtype=o3c.Dtype.UInt8)
        
        # OccMap.logger.info(f"===========================")
        # OccMap.logger.info(f"Colored point cloud with {len(colors)} points")
        # OccMap.logger.info(f"Using color: RGB{tuple(rgb_color)}")
        # OccMap.logger.info(f"===========================\n")
        
        return colored_pcd

    @staticmethod
    def downsample_pcd_by_class(pcd: o3d.t.geometry.PointCloud,
                                classes_to_downsample: List[int] = None,
                                voxel_size: float = 0.01) -> o3d.t.geometry.PointCloud:
        """Downsample points in a point cloud, optionally only downsampling specific classes.
        
        Args:
            pcd: Input point cloud to downsample
            classes_to_downsample: List of class labels to downsample. If None, downsample all points
            voxel_size: Size of voxels for downsampling
            
        Returns:
            o3d.t.geometry.PointCloud: Downsampled point cloud
        """
        # get the labels from the pcd
        labels = pcd.point['label'].numpy()

        if classes_to_downsample is None:
            # if no classes to downsample, downsample all points
            downsampled_points = pcd.voxel_down_sample(voxel_size)
            downsampled_pcd = downsampled_points
        else:
            # create a mask for points that should be downsampled
            label_mask = np.isin(labels, classes_to_downsample)
            
            # downsample only the points with the specified labels
            points_to_downsample = pcd.select_by_mask(o3c.Tensor(label_mask.ravel()))
            downsampled_points = points_to_downsample.voxel_down_sample(voxel_size)
            
            # create a mask for points that should not be downsampled
            mask_not_downsample = ~label_mask
            points_not_to_downsample = pcd.select_by_mask(o3c.Tensor(mask_not_downsample.ravel()))
            
            # combine the downsampled points with the points that were not downsampled
            downsampled_pcd = OccMap.combine_pcds([downsampled_points, points_not_to_downsample])
            
        return downsampled_pcd

    @staticmethod
    def filter_near_points(visible_pcd: o3d.t.geometry.PointCloud,
                           hidden_pcd: o3d.t.geometry.PointCloud,
                           radius: float = 0.01) -> o3d.t.geometry.PointCloud:
        """Filter visible_pcd points that are near to hidden_pcd points.
        
        Args:
            visible_pcd: Point cloud containing visible points
            hidden_pcd: Point cloud containing hidden points
            radius: Minimum distance from hidden points (default: 0.01)
            
        Returns:
            o3d.t.geometry.PointCloud: Filtered point cloud
        """
        
        # if no hidden points, return visible_pcd
        hidden_points = hidden_pcd.point['positions'].numpy()
        if len(hidden_points) == 0:
            return visible_pcd
            
        # convert to legacy point cloud for nearest neighbor operations
        hidden_legacy = o3d.geometry.PointCloud()
        hidden_legacy.points = o3d.utility.Vector3dVector(hidden_points)
        
        visible_legacy = o3d.geometry.PointCloud()
        visible_legacy.points = o3d.utility.Vector3dVector(
            visible_pcd.point['positions'].numpy())
        
        # find points that are far enough from hidden points
        distances = np.asarray(visible_legacy.compute_point_cloud_distance(hidden_legacy))
        far_mask = distances >= radius
        
        # filter points
        filtered_pcd = o3d.t.geometry.PointCloud()
        filtered_pcd.point['positions'] = o3c.Tensor(
            visible_pcd.point['positions'].numpy()[far_mask])
        filtered_pcd.point['colors'] = o3c.Tensor(
            visible_pcd.point['colors'].numpy()[far_mask])
            
        return filtered_pcd

    @staticmethod
    def get_stereo_occ_pcd(sfm_pcd: o3d.t.geometry.PointCloud,
                           stereo_pcd: o3d.t.geometry.PointCloud,
                           K: np.ndarray,
                           P: np.ndarray,
                           to_crop: bool = True,
                           bb: Dict[str, float] = None,
                           classes_to_downsample: List[int] = [3],
                           voxel_size: float = 0.01,
                           img_shape: Tuple[int, int] = (1080, 1920), 
                           logging_level: int = 0) -> o3d.t.geometry.PointCloud:
        """Generate point cloud with occluded points colored using both SfM and stereo point clouds.
        
        Args:
            sfm_pcd: Input SfM point cloud
            stereo_pcd: Input stereo point cloud
            K: 3x3 camera intrinsic matrix
            P: 3x4 camera extrinsic matrix
            bb: Optional bounding box for cropping
            classes_to_downsample: List of class labels to downsample. If None, downsample all points
            voxel_size: Size of voxels for downsampling
            to_crop: Whether to crop the point clouds
            img_shape: Tuple of (height, width) for projection
            
        Returns:
            o3d.t.geometry.PointCloud: Point cloud with occluded points colored
        """
        assert K is not None and K.shape == (3, 3), "camera_intrinsic_matrix must be a 3x3 matrix"
        assert P is not None and P.shape == (3,4), "camera_extrinsic_matrix must be a 3x4 matrix"
        assert img_shape == (1080, 1920), "img_shape must be (1080, 1920)"
        assert bb is not None if to_crop else True, "bounding box must be provided"
        
        logger = get_logger("occ_mask_generator", logging_level)

        if P is None:
            P = np.hstack([np.eye(3), np.zeros((3,1))])

        # crop point clouds if requested
        sfm_pcd_cropped = crop_pcd(sfm_pcd, bb) if to_crop else sfm_pcd
        stereo_pcd_cropped = crop_pcd(stereo_pcd, bb) if to_crop else stereo_pcd
        
        img_coords = OccMap.get_pixel_coordinates(sfm_pcd_cropped, K, P)
        sfm_u , sfm_v = img_coords[:, 0], img_coords[:, 1]
        sfm_depths = sfm_pcd_cropped.point['positions'].numpy()[:, 2]
        
        depth_buffer: np.ndarray = OccMap.generate_depth_buffer(sfm_pcd_cropped, stereo_pcd_cropped, K = K, P = P)
        h, w = img_shape

        sfm_hidden_mask: np.ndarray = np.zeros(len(sfm_depths), dtype=bool)
        sfm_bound_mask: np.ndarray = np.zeros(len(sfm_depths), dtype=bool)

        assert len(sfm_u) == len(sfm_v) == len(sfm_depths), \
            f"sfm_u, sfm_v, and sfm_depths must have the same length"

        for i, (u, v, d) in enumerate(zip(sfm_u, sfm_v, sfm_depths)):
            if 0 <= u < w and 0 <= v < h:
                if d > depth_buffer[v, u] + OccMap.DEPTH_THRESHOLD:
                    sfm_hidden_mask[i] = True
            else:
                sfm_bound_mask[i] = True

        
        positions = sfm_pcd_cropped.point['positions'].numpy()
        colors = sfm_pcd_cropped.point['colors'].numpy()
        labels = sfm_pcd_cropped.point['label'].numpy()

        # get indices of points that don't belong to either mask
        visible_mask = ~(sfm_hidden_mask | sfm_bound_mask)

        # separate pcds for hidden, bound, and visible points
        hidden_pcd = o3d.t.geometry.PointCloud()
        bound_pcd = o3d.t.geometry.PointCloud()
        visible_pcd = o3d.t.geometry.PointCloud()
        
        # create hidden_pcd 
        hidden_pcd.point['positions'] = o3c.Tensor(positions[sfm_hidden_mask])
        hidden_pcd.point['colors'] = o3c.Tensor(colors[sfm_hidden_mask])
        
        # create bound_pcd
        bound_pcd.point['positions'] = o3c.Tensor(positions[sfm_bound_mask])
        bound_pcd.point['colors'] = o3c.Tensor(colors[sfm_bound_mask])

        # create visible_pcd
        visible_pcd.point['positions'] = o3c.Tensor(positions[visible_mask])
        visible_pcd.point['colors'] = o3c.Tensor(colors[visible_mask])
        visible_pcd.point['label'] = o3c.Tensor(labels[visible_mask])
        
        # downsample vine_canopy points
        visible_downsampled = OccMap.downsample_pcd_by_class(visible_pcd, 
                                                             classes_to_downsample=classes_to_downsample,
                                                             voxel_size=voxel_size)

        # filter visible points that are near to hidden points
        visible_filtered = OccMap.filter_near_points(visible_downsampled, 
                                                     hidden_pcd, 
                                                     radius=0.01)

       
        # # filter visible points that are near to hidden points
        # visible_filtered = OccMap.filter_near_points(visible_pcd, 
        #                                              hidden_pcd, 
        #                                              radius=0.01)


        # visible_downsampled = OccMap.downsample_pcd_by_class(visible_pcd, 
        #                                                           voxel_size=0.01)

        # visible_filtered = OccMap.filter_near_points(visible_downsampled, hidden_pcd, 
        #                                       radius=0.01)

        final_pcd = OccMap.combine_pcds([hidden_pcd, bound_pcd, visible_filtered])
        
        total_points = len(final_pcd.point['positions'])
        num_hidden = len(hidden_pcd.point['positions'])
        num_bound = len(bound_pcd.point['positions'])

        
        # re-create masks for hidden and bound points
        new_hidden_mask = np.zeros(total_points, dtype=bool)
        new_bound_mask = np.zeros(total_points, dtype=bool)

        # now we can safely set the masks knowing the exact order
        new_hidden_mask[:num_hidden] = True
        new_bound_mask[num_hidden:num_hidden + num_bound] = True

        logger.info(f"===========================")
        logger.info(f"STEREO OCC-MASK GENERATION:")
        logger.info(f"- total points: {total_points}")
        logger.info(f"- hidden points: {new_hidden_mask.sum()}")
        logger.info(f"- bound points: {new_bound_mask.sum()}")
        logger.info(f"- % hidden: {100 * new_hidden_mask.sum() / total_points:.2f}%")
        logger.info(f"- % bound: {100 * new_bound_mask.sum() / total_points:.2f}%")
        logger.info(f"- % occ: {100 * (new_hidden_mask.sum() + new_bound_mask.sum()) / total_points:.2f}%")
        logger.info(f"===========================\n")
        
        # color points using the new method
        hidden_color = np.array([241, 196, 15])     # sun yellow
        bound_color = np.array([142, 68, 173])   # wisteria purple
        
        OccMap.add_mask_color_to_pcd(final_pcd, [new_hidden_mask, new_bound_mask], 
                                    [hidden_color, bound_color])

        return final_pcd