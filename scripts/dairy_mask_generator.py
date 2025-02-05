#! /usr/bin/env python3
import logging, coloredlogs
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import open3d.core as o3c
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional, Dict, Union
import yaml

from scripts.helpers import crop_pcd, mono_to_rgb_mask
from scripts.logger import get_logger


class RotationUtils:
    def __init__(self):
        pass    

    @staticmethod
    def ypr_to_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
        '''
        Convert yaw, pitch, roll angles in degrees to a 3x3 rotation matrix.
        Follows the ZYX convention (yaw → pitch → roll)
        
        Args:
            yaw (float): rotation around Z axis in degrees
            pitch (float): rotation around Y axis in degrees
            roll (float): rotation around X axis in degrees
            
        Returns:
            np.ndarray: 3x3 rotation matrix
        '''
        # Convert angles to radians
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        roll_rad = np.radians(roll)

        # Individual rotation matrices
        # Rz (yaw) - rotation around Z axis
        Rz = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                      [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                      [0, 0, 1]])

        # Ry (pitch) - rotation around Y axis
        Ry = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                      [0, 1, 0],
                      [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])

        # Rx (roll) - rotation around X axis
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(roll_rad), -np.sin(roll_rad)],
                      [0, np.sin(roll_rad), np.cos(roll_rad)]])

        # Combined rotation matrix R = Rz * Ry * Rx (ZYX convention)
        R = Rz @ Ry @ Rx
        return R

    @staticmethod
    def rotation_matrix_to_ypr(R: np.ndarray) -> Tuple[float, float, float]:
        '''
        Convert rotation matrix to yaw, pitch, roll angles using ZYX convention.
        
        Args:
            R (np.ndarray): 3x3 rotation matrix
            
        Returns:
            Tuple[float, float, float]: yaw, pitch, roll angles in degrees
        '''
        # Ensure proper matrix shape
        if R.shape != (3, 3):
            raise ValueError("Input matrix must be 3x3")

        # Extract angles using ZYX convention
        # Handle singularity when cos(pitch) is close to zero
        cos_pitch = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = cos_pitch < 1e-6

        if not singular:
            yaw = np.arctan2(R[1, 0], R[0, 0])  # atan2(sin(yaw)cos(pitch), cos(yaw)cos(pitch))
            pitch = np.arctan2(-R[2, 0], cos_pitch)  # atan2(-sin(pitch), cos(pitch))
            roll = np.arctan2(R[2, 1], R[2, 2])  # atan2(sin(roll)cos(pitch), cos(roll)cos(pitch))
        else:
            # Gimbal lock case (pitch = ±90°)
            yaw = np.arctan2(-R[1, 2], R[1, 1])  # arbitrary choice
            pitch = np.arctan2(-R[2, 0], cos_pitch)  # ±90°
            roll = 0  # arbitrary choice

        return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

    @staticmethod
    def rotation_matrix_to_axis_angles(R: np.ndarray) -> Tuple[float, float, float]:
        '''
        Convert 3x3 rotation matrix to angles between rotated and original coordinate axes.
        
        Args:
            R (np.ndarray): 3x3 rotation matrix
            
        Returns:
            Tuple[float, float, float]: angles (in degrees) between original and rotated 
                                       x, y, and z axes respectively
        '''
        if R.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")
        
        # Original coordinate axes
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])
        
        # Rotated coordinate axes
        rotated_x = R @ x_axis
        rotated_y = R @ y_axis
        rotated_z = R @ z_axis
        
        # Calculate angles between original and rotated axes
        # Using np.clip to handle numerical precision issues
        angle_x = np.arccos(np.clip(np.dot(x_axis, rotated_x), -1.0, 1.0))
        angle_y = np.arccos(np.clip(np.dot(y_axis, rotated_y), -1.0, 1.0))
        angle_z = np.arccos(np.clip(np.dot(z_axis, rotated_z), -1.0, 1.0))
        
        return np.degrees(angle_x), np.degrees(angle_y), np.degrees(angle_z)

    @staticmethod
    def verify_rotation_conversion(R: np.ndarray, tolerance: float = 1e-10) -> bool:
        '''
        Verify that rotation matrix conversions are consistent.
        
        Args:
            R (np.ndarray): Original rotation matrix
            tolerance (float): Maximum allowed difference between matrices
            
        Returns:
            bool: True if conversion is consistent within tolerance
        '''
        # Convert to YPR and back to rotation matrix
        yaw, pitch, roll = RotationUtils.rotation_matrix_to_ypr(R)
        R_reconstructed = RotationUtils.ypr_to_rotation_matrix(yaw, pitch, roll)
        
        # Check if matrices are similar within tolerance
        diff = np.abs(R - R_reconstructed).max()
        return diff <= tolerance


class BEVGenerator:
    def __init__(self, logging_level=logging.WARNING, yaml_path: Optional[str] = None):
        '''
        BEV data from segmented pointcloud
        
        Args:
            logging_level: Logging level (default: logging.WARNING)
            yaml_path: Optional path to yaml file containing label configurations
        '''
        assert yaml_path is not None, "yaml_path is required!"
        
        self.logger = get_logger("bev_generator", level=logging_level)
        
        self.logger.info(f"=================================")    
        self.logger.info(f"BEVGenerator initialized")
        self.logger.info(f"=================================\n")

        # define custom priority mapping (lower number = higher priority)
        priority_mapping = {
            'OBSTACLE': 1,
            'COWS': 2,
            'FENCE': 3,
            'CATTLE_GUARD': 4,
            'POLES': 5,
            'GATES': 6,
            'FEED': 7,
            'FEED_PUSHER': 8,
            'VEGETATION': 9,
            'NAVIGABLE_SPACE': 10,
            
        }
        
        self.yaml_path = yaml_path
        self.LABELS = self.update_labels_from_yaml(yaml_path, priority_mapping)
        # self.log_label_info()
        
        # tilt-correction matrix
        self.R = None

        # old / rectified ground plane normal
        self.old_normal = None  
        self.new_normal = None

        # ground plane inliers
        self.ground_inliers = None

        # rectified 3D BEV pcd
        self.pcd_BEV_3D = None

        # rectified PCD
        self.pcd_RECTIFIED = None


    def count_unique_labels(self, pcd: o3d.t.geometry.PointCloud) -> Tuple[int, np.ndarray, Dict[int, int]]:
        """
        Count the number and frequency of unique labels in a point cloud.

        Args:
            pcd (o3d.t.geometry.PointCloud): The point cloud from which to count unique labels.

        Returns:
            Tuple[int, np.ndarray, Dict[int, int]]: A tuple containing:
                - The number of unique labels in the point cloud
                - Array of unique label values (empty if no labels)
                - Dictionary mapping label IDs to their counts
        """
        # check if 'label' attribute exists and has data
        if 'label' not in pcd.point or pcd.point['label'].shape[0] == 0:
            self.logger.warning("The point cloud does not contain any labels.")
            return 0, np.array([]), {}
        
        labels = pcd.point['label'].numpy()
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_points = len(labels)
        num_unique_labels = len(unique_labels)
        
        # create dictionary mapping labels to counts
        label_counts = dict(zip(unique_labels, counts))
        
        # log the counts and percentages
        self.logger.info(f"=================================")
        self.logger.info(f"Label distribution in point cloud:")
        for label_id, count in label_counts.items():
            label_name = next((k for k, v in self.LABELS.items() if v['id'] == label_id), f"Unknown-{label_id}")
            percentage = (count / total_points) * 100
            self.logger.info(f"{label_name:<15}: {count:>7} points ({percentage:>6.2f}%)")
        self.logger.info(f"Total points: {total_points}")
        self.logger.info(f"Unique labels: {num_unique_labels}")
        self.logger.info(f"=================================\n")
        
        return num_unique_labels, unique_labels, label_counts


    def filter_radius_outliers(self, pcd: o3d.t.geometry.PointCloud, nb_points: int, search_radius: float):
        '''
        Filter radius-based outliers from the point cloud
        '''
        _, ind = pcd.remove_radius_outliers(nb_points=nb_points, search_radius=search_radius)
        inliers = pcd.select_by_mask(ind)
        outliers = pcd.select_by_mask(ind, invert=True)
        return inliers, outliers

    def get_plane_model(self,pcd: o3d.t.geometry.PointCloud, class_label: int):
        '''
        returns [a,b,c,d]
        '''
        pcd_class = self.get_class_pointcloud(pcd, class_label)
        plane_model, inliers = pcd_class.segment_plane(distance_threshold=0.01,
                                                        ransac_n=3,
                                                        num_iterations=1000)
        return plane_model.numpy()
    
    def align_normal_to_y_axis(self,normal_):
        '''
        Rotation matrix to align the normal vector to the y-axis
        '''
        y_axis = np.array([0, 1, 0])
        v = np.cross(normal_, y_axis)
        s = np.linalg.norm(v)
        c = np.dot(normal_, y_axis)
        I = np.eye(3)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = I + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
        return R
    
    def merge_pcds(self, bev_collection: List[o3d.t.geometry.PointCloud]) -> o3d.t.geometry.PointCloud:
        '''
        Merge collection of pcds along with their labels / colors / points
        '''
        
        # stack positions
        position_tensors: List[np.ndarray] = [pcd.point['positions'].numpy() for pcd in bev_collection]
        stacked_positions: o3c.Tensor = o3c.Tensor(np.vstack(position_tensors), dtype=o3c.Dtype.Float32)
        
        # stack labels
        label_tensors: List[np.ndarray] = [pcd.point['label'].numpy() for pcd in bev_collection]
        stacked_labels: o3c.Tensor = o3c.Tensor(np.vstack(label_tensors), dtype=o3c.Dtype.UInt8)
        
        # stack colors
        color_tensors: List[np.ndarray] = [pcd.point['colors'].numpy() for pcd in bev_collection]
        stacked_colors: o3c.Tensor = o3c.Tensor(np.vstack(color_tensors), dtype=o3c.Dtype.UInt8)

        # Create a unified point cloud
        map_to_tensors: Dict[str, o3c.Tensor] = {
            'positions': stacked_positions,
            'label': stacked_labels,
            'colors': stacked_colors        
        }

        # Create a unified point cloud
        combined_pcd: o3d.t.geometry.PointCloud = o3d.t.geometry.PointCloud(map_to_tensors)    
        return combined_pcd
    
    def axis_angles(self,vec):
        '''
        Calculate the angles between input vector and the coordinate axes
        '''
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])
        
        angle_x = np.arccos(np.dot(vec, x_axis) / np.linalg.norm(vec))
        angle_y = np.arccos(np.dot(vec, y_axis) / np.linalg.norm(vec))
        angle_z = np.arccos(np.dot(vec, z_axis) / np.linalg.norm(vec))
        
        return np.degrees(angle_x), np.degrees(angle_y), np.degrees(angle_z)

    def compute_tilt_matrix(self, pcd: o3d.t.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Compute navigation-space tilt w.r.t y-axis
        '''
        ground_normal, ground_inliers = self.get_class_plane(pcd, self.LABELS["NAVIGABLE_SPACE"]["id"])
        R = self.align_normal_to_y_axis(ground_normal)
        return R, ground_normal, ground_inliers

    def get_class_pointcloud(self, pcd: o3d.t.geometry.PointCloud, class_label: int):
        '''
        Returns class-specific point cloud
        '''
        mask = pcd.point["label"] == class_label
        pcd_labels = pcd.select_by_index(mask.nonzero()[0])
        return pcd_labels

    def get_class_plane(self,pcd: o3d.t.geometry.PointCloud, class_label: int) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Get the inliers / normal vector for the labelled pointcloud
        '''
        pcd_class = self.get_class_pointcloud(pcd, class_label)
        plane_model, inliers = pcd_class.segment_plane(distance_threshold=0.01,
                                                        ransac_n=3,
                                                        num_iterations=1000)
        [a, b, c, d] = plane_model.numpy()
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal) 
        return normal, inliers
    
    def get_tilt_rectified_pcd(self, pcd_input: o3d.t.geometry.PointCloud) -> o3d.t.geometry.PointCloud:
        '''
        The function aligns the GROUND plane normal to CAMERA y-axis \
        in the CAMERA frame of reference
        '''

        if self.pcd_RECTIFIED is not None:
            self.logger.warning(f"=================================")      
            self.logger.warning(f"Rectified pcd already computed!")
            self.logger.warning(f"=================================\n")
            return self.pcd_RECTIFIED

        R, old_normal, ground_inliers = self.compute_tilt_matrix(pcd_input)
        
        # updating class variables
        self.R = R
        self.old_normal = old_normal
        self.ground_inliers = ground_inliers
        
        # [before tilt rectification]
        # angle made by the normal vector with the [x-axis, y-axis, z-axis]
        axis_x, axis_y, axis_z = RotationUtils.rotation_matrix_to_axis_angles(R)
        
        self.logger.info(f"=================================")      
        self.logger.info(f"BEFORE TILT RECTIFICATION...")
        self.logger.info(f"Ground plane normal makes [{axis_x:.2f},{axis_y:.2f},{axis_z:.2f}] degrees"
                        " with the [x-axis, y-axis, z-axis] respectively!")
        self.logger.info(f"=================================\n")

        
        new_normal = np.dot(self.old_normal, R.T)
        self.new_normal = new_normal

        # [after tilt rectification]
        # angle made by the normal vector with the [x-axis, y-axis, z-axis]
        angles = self.axis_angles(self.new_normal)
        
        self.logger.info(f"=================================")    
        self.logger.info(f"AFTER TILT RECTIFICATION...")
        self.logger.info(f"Ground plane normal makes [{angles[0]:.2f},{angles[1]:.2f},{angles[2]:.2f}] degrees"
                            " with the [x-axis, y-axis, z-axis] respectively!")
        self.logger.info(f"=================================\n")
    
        # angle between normal and y-axis should be close to 0 degrees
        assert np.isclose(angles[1], 0, atol=1e-10), f"Error: angles_transformed[1] is {angles[1]}"\
            "but should be close to 0 degrees. Check tilt correction!"


        pcd_corrected = pcd_input.clone()
        
        # making y-axis perpendicular to the ground plane + right-handed coordinate system
        pcd_corrected.rotate(R, center=(0, 0, 0))

        # updating class variables
        self.pcd_RECTIFIED = pcd_corrected
        return pcd_corrected
    
    def get_normal_alignment(self):
        '''
        Get the angle between the new normal and the y-axis
        '''
        angles = self.axis_angles(self.new_normal)
        return angles[1]

    def get_pcd_BEV_3D(self) -> o3d.t.geometry.PointCloud:
        '''Get the merged 3D BEV pointcloud just before bev generation'''
        assert self.pcd_BEV_3D is not None, "3D BEV pointcloud is not initialized!"
        return self.pcd_BEV_3D

    def generate_pcd_BEV_2D(self, pcd_input: o3d.t.geometry.PointCloud) -> o3d.t.geometry.PointCloud:
        
        """Generate BEV pcd (with z = 0) from segmented pointcloud"""
        
        self.logger.info(f"=================================")    
        self.logger.info(f"Generating BEV pcd...")
        self.logger.info(f"=================================\n")

        pcd_RECTIFIED: o3d.t.geometry.PointCloud = self.get_tilt_rectified_pcd(pcd_input)

        pcd_OBSTACLE: o3d.t.geometry.PointCloud = self.get_class_pointcloud(pcd_RECTIFIED, self.LABELS["OBSTACLE"]["id"])
        pcd_FEED: o3d.t.geometry.PointCloud = self.get_class_pointcloud(pcd_RECTIFIED, self.LABELS["FEED"]["id"])
        pcd_FENCE: o3d.t.geometry.PointCloud = self.get_class_pointcloud(pcd_RECTIFIED, self.LABELS["FENCE"]["id"])
        pcd_VEGETATION: o3d.t.geometry.PointCloud = self.get_class_pointcloud(pcd_RECTIFIED, self.LABELS["VEGETATION"]["id"])
        pcd_COWS: o3d.t.geometry.PointCloud = self.get_class_pointcloud(pcd_RECTIFIED, self.LABELS["COWS"]["id"])
        pcd_GATES: o3d.t.geometry.PointCloud = self.get_class_pointcloud(pcd_RECTIFIED, self.LABELS["GATES"]["id"])
        pcd_POLES: o3d.t.geometry.PointCloud = self.get_class_pointcloud(pcd_RECTIFIED, self.LABELS["POLES"]["id"])
        
        # using only ground-inliers
        pcd_NAVIGABLE: o3d.t.geometry.PointCloud = self.get_class_pointcloud(pcd_RECTIFIED, self.LABELS["NAVIGABLE_SPACE"]["id"])
        # pcd_NAVIGABLE = pcd_NAVIGABLE.select_by_index(self.ground_inliers)

        
        

        # merging label-wise pointclouds
        self.pcd_BEV_3D: o3d.t.geometry.PointCloud = self.merge_pcds([pcd_OBSTACLE,
                                                                      pcd_FEED,
                                                                      pcd_FENCE,
                                                                      pcd_VEGETATION,
                                                                      pcd_COWS,
                                                                      pcd_GATES, 
                                                                      pcd_POLES, 
                                                                      pcd_NAVIGABLE])
        # converting to BEV
        pcd_BEV_2D: o3d.t.geometry.PointCloud = self.pcd_BEV_3D.clone()
        pcd_BEV_2D.point['positions'][:, 1] = 0.0  # Set all y-coordinates to 0
        return pcd_BEV_2D
    

    def bev_pcd_to_seg_mask_mono(self, pcd: o3d.t.geometry.PointCloud, 
                                      nx: int = 200, nz: int = 200, 
                                      bb: dict = None) -> np.ndarray:
        """
        Generate a 2D single-channel segmentation mask from a BEV point cloud.

        :param pcd: Open3D tensor point cloud with labels
        :param nx: Number of grid cells along the x-axis (horizontal)
        :param nz: Number of grid cells along the z-axis (vertical)
        :param bb: Bounding box parameters as a dictionary {'x_min', 'x_max', 'z_min', 'z_max'}
        :return: Segmentation mask as a 2D numpy array
        """
        
        assert bb is not None, "Bounding box parameters are required!"
        assert nx is not None and nz is not None, "nx and nz must be provided!"
        assert nx == nz, "nx and nz must be equal!"

        self.logger.info(f"=================================")    
        self.logger.info(f"(nx, nz): ({nx}, {nz})")
        self.logger.info(f"=================================\n")
        
        # Extract bounding box limits
        x_min, x_max = bb['x_min'], bb['x_max']
        z_min, z_max = bb['z_min'], bb['z_max']

        # Calculate grid resolution
        res_x: float = (x_max - x_min) / nx
        res_z: float = (z_max - z_min) / nz
        
        self.logger.info(f"=================================")    
        self.logger.info(f"(res_x, res_z): ({res_x:.4f}, {res_z:.4f})")
        self.logger.info(f"=================================\n")
        
        assert res_x == res_z, "Resolution x and z must be equal!"

        self.logger.info(f"=================================")    
        self.logger.info(f"Resolution X: {res_x:.4f} meters, Z: {res_z:.4f} meters")
        self.logger.info(f"=================================\n")

        # extract point coordinates and labels
        x_coords: np.ndarray = pcd.point['positions'][:, 0].numpy()
        z_coords: np.ndarray = pcd.point['positions'][:, 2].numpy()
        labels: np.ndarray = pcd.point['label'].numpy()

        # generate mask_x and mask_z using res_x
        mask_x: np.ndarray = ((x_coords - x_min) / res_x).astype(np.int32)
        assert mask_x.min() >= 0 and mask_x.max() < nx, "x-indices are out of bounds!"
        
        mask_z: np.ndarray = nz - 1 - ((z_coords - z_min) / res_z).astype(np.int32)
        assert mask_z.min() >= 0 and mask_z.max() < nz, "z-indices are out of bounds!"
        
        # initialize mask with maximum priority value (lowest priority)
        mask: np.ndarray = np.full((nz, nx), 255, dtype=np.uint8)
        
        # Create priority lookup for each label
        label_priorities: Dict[int, int] = {label_info["id"]: label_info["priority"] 
                          for label_info in self.LABELS.values()}
        
     
        labels: np.ndarray = pcd.point['label'].numpy().astype(np.int32).flatten()
        
        # Assign labels respecting priorities (lower priority number = higher priority)
        for x, z, label in zip(mask_x, mask_z, labels):
            current_priority: int = label_priorities.get(int(mask[z, x]), float('inf'))
            new_priority: int = label_priorities.get(int(label), float('inf'))
            
            if new_priority < current_priority:
                mask[z, x] = label
        
        return mask

    def pcd_to_seg_mask(self, 
                             pcd: o3d.t.geometry.PointCloud, 
                             nx: int = None, nz: int = None, 
                             bb: dict = None, 
                             ) -> Tuple[np.ndarray, np.ndarray]:
        
        '''Generate mono / rgb segmentation masks from a pointcloud'''
                
        assert bb is not None, "Bounding box parameters are required!"
        assert nx is not None and nz is not None, "nx and nz must be provided!"
        assert nx == nz, "nx and nz must be equal!"
        assert self.yaml_path is not None, "yaml_path is required!"
        
        self.logger.info(f"=================================")    
        self.logger.info(f"Generating segmentation mask...")
        self.logger.info(f"self.yaml_path: {self.yaml_path}")
        self.logger.info(f"=================================\n")


        pcd_BEV_2D = self.generate_pcd_BEV_2D(pcd)
        pcd_BEV_2D_CROPPED = crop_pcd(pcd_BEV_2D, bb)

        seg_mask_mono = self.bev_pcd_to_seg_mask_mono(pcd_BEV_2D_CROPPED, nx, nz, bb)
        seg_mask_rgb = mono_to_rgb_mask(mono_mask=seg_mask_mono, yaml_path=self.yaml_path)

        return seg_mask_mono, seg_mask_rgb

    def get_updated_camera_extrinsics(self) -> np.ndarray:
        '''Get updated camera extrinsics after tilting the pointcloud'''
        
        assert self.R is not None, "Rotation matrix is required!"
        T = np.eye(4)
        R = self.R.T
        T[:3, :3] = R
        return T

    def update_labels_from_yaml(self, yaml_path: str, 
                              priority_mapping: Dict[str, int]
                              ) -> Dict[str, Dict[str, Union[int, str, List[int]]]]:
        """
        Update LABELS dictionary using configuration from yaml file.
        
        Args:
            yaml_path (str): Path to the yaml configuration file
            priority_mapping (Dict[str, int]): Mapping of label keys to their priorities.
                Only labels with defined priorities will be included in the output
            
        Note:
            The yaml file should contain 'labels' and 'color_map' entries
            Labels without defined priorities will be skipped
            
        Raises:
            ValueError: If priority_mapping is None
        """
        # verify priority mapping is provided
        if priority_mapping is None:
            raise ValueError("Priority mapping must be provided")
            
        try:
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)
                
            # create mapping from label names to their properties
            label_mapping = {}
            skipped_labels = []
            
            for label_id, label_name in config['labels'].items():
                # skip void class
                if label_id == 0:
                    continue
                    
                # convert label name to uppercase and replace spaces/underscores with underscore
                label_key = label_name.upper().replace(' ', '_').replace('-', '_')
                
                # skip labels without defined priorities
                if label_key not in priority_mapping:
                    skipped_labels.append(label_key)
                    continue
                
                # get color from color map (keep as BGR)
                bgr_color = config['color_map'][label_id]
                
                # get priority from mapping
                priority = priority_mapping[label_key]
                
                label_mapping[label_key] = {
                    "id": label_id,
                    "priority": priority,
                    "name": label_name,
                }
                
            # log update information
            # self.logger.info(f"=================================")
            # self.logger.info(f"Updated LABELS from {yaml_path}")
            # self.logger.info(f"Number of labels included: {len(label_mapping)}")
            # self.logger.info(f"Priority mapping applied: {priority_mapping}")
            if skipped_labels:
                self.logger.info(f"=================================")
                self.logger.info(f"Skipped labels (no priority defined): {skipped_labels}")
                self.logger.info(f"=================================\n")
            
            return label_mapping

        except Exception as e:
            self.logger.error(f"Failed to update labels from {yaml_path}: {str(e)}")
            raise

    def log_label_info(self) -> None:
        """
        Log information about configured labels in a structured format.
        Creates a visually organized table of label information including ID, priority, and name.
        """
        # create a visual separator for better log readability
        separator = "=" * 50
        self.logger.warning(separator)
        
        for key, value in self.LABELS.items():
            label_info = (
                f"Label: {key:15} | "  
                f"ID: {value['id']:<3} | "
                f"Priority: {value['priority']:<2} | "
                f"Name: {value['name']}"
            )
            self.logger.warning(label_info)
            
        self.logger.warning(separator + "\n")

  