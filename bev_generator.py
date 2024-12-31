#! /usr/bin/env python3
import logging, coloredlogs
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import open3d.core as o3c
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional
import torch

from helpers import crop_pcd, mono_to_rgb_mask
from logger import get_logger


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
    def __init__(self):
        
        '''BEV data from segmented pointcloud'''
        
        self.LABELS = {    
            "OBSTACLE": {"id": 1, "priority": 1},
            "VINE_POLE": {"id": 5, "priority": 2},  
            "VINE_CANOPY": {"id": 3, "priority": 3},
            "VINE_STEM": {"id": 4, "priority": 4},  
            "NAVIGABLE_SPACE": {"id": 2, "priority": 5},  
        }
        
        self.R = None
        self.logger = get_logger("bev_generator", level=logging.WARNING)

        self.logger.info(f"=================================")      
        self.logger.info(f"BEVGenerator initialized")
        self.logger.info(f"=================================\n")
    
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
    
    def generate_unified_bev_pcd(self, bev_collection):
        '''
        Generate a unified BEV pointcloud from a collection of label-specific BEV pointclouds
        '''
        # Vertically stack the point positions of the bev_collection pointclouds
        position_tensors = [pcd.point['positions'].numpy() for pcd in bev_collection]
        stacked_positions = o3c.Tensor(np.vstack(position_tensors), dtype=o3c.Dtype.Float32)
        
        # Vertically stack the point labels of the bev_collection pointclouds
        label_tensors = [pcd.point['label'].numpy() for pcd in bev_collection]
        stacked_labels = o3c.Tensor(np.vstack(label_tensors), dtype=o3c.Dtype.Int32)

        # Vertically stack the point colors of the bev_collection pointclouds
        color_tensors = [pcd.point['colors'].numpy() for pcd in bev_collection]
        stacked_colors = o3c.Tensor(np.vstack(color_tensors), dtype=o3c.Dtype.UInt8)
        
        # Create a unified point cloud
        map_to_tensors = {}
        map_to_tensors['positions'] = stacked_positions
        map_to_tensors['label'] = stacked_labels
        map_to_tensors['colors'] = stacked_colors        

        combined_pcd = o3d.t.geometry.PointCloud(map_to_tensors)    
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

    def compute_tilt_matrix(self, pcd: o3d.t.geometry.PointCloud):
        '''
        Compute navigation-space tilt w.r.t y-axis
        '''
        normal, _ = self.get_class_plane(pcd, self.LABELS["NAVIGABLE_SPACE"]["id"])
        R = self.align_normal_to_y_axis(normal)
        return R

    def get_class_pointcloud(self, pcd: o3d.t.geometry.PointCloud, class_label: int):
        '''
        Returns class-specific point cloud
        '''
        mask = pcd.point["label"] == class_label
        pcd_labels = pcd.select_by_index(mask.nonzero()[0])
        return pcd_labels

    def get_class_plane(self,pcd: o3d.t.geometry.PointCloud, class_label: int):
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
    


    def clean_around_label(
        self, 
        pcd_target: o3d.t.geometry.PointCloud, 
        pcd_source: o3d.t.geometry.PointCloud, 
        tolerance: float = 0.02
    ) -> o3d.t.geometry.PointCloud:
        
        '''
        Remove points in pcd_source that are close to the points in pcd_target
        '''
        
        # Get positions for stem and canopy
        target_points = pcd_target.point['positions'].numpy()
        source_points = pcd_source.point['positions'].numpy()

        # Create arrays of (x,z) coordinates
        target_xz = target_points[:, [0,2]]  # Get x,z columns
        source_xz = source_points[:, [0,2]]  # Get x,z columns

        # Create KD-trees for efficient nearest neighbor search
        target_tree = cKDTree(target_xz)
        source_tree = cKDTree(source_xz)    

        # Find matches within a tolerance of 0.01
        tolerance = 0.02
        matches = []

        to_remove = []
        for i in range(len(source_xz)):
            indices = target_tree.query_ball_point(source_xz[i], r=tolerance)
            if len(indices) > 0:
                to_remove.append(i)
        
        positions = np.delete(pcd_source.point['positions'].numpy(), to_remove, axis=0)
        labels = np.delete(pcd_source.point['label'].numpy(), to_remove, axis=0)
        colors = np.delete(pcd_source.point['colors'].numpy(), to_remove, axis=0)
        
        # Update bev_navigable with new positions and labels
        pcd_source.point['positions'] = o3c.Tensor(positions, dtype=o3c.Dtype.Float32)
        pcd_source.point['label'] = o3c.Tensor(labels, dtype=o3c.Dtype.Int32)
        pcd_source.point['colors'] = o3c.Tensor(colors, dtype=o3c.Dtype.UInt8)

        # self.logger.info(f"=================================")
        # self.logger.info(f"Found {len(to_remove)} matches!")
        # self.logger.info(f"=================================\n")

        return pcd_source

    def clean_around_labels(
        self, 
        pcd_target: o3d.t.geometry.PointCloud, 
        pcd_sources: List[o3d.t.geometry.PointCloud], 
        tolerance: float = 0.02
    ) -> List[o3d.t.geometry.PointCloud]:
        
        '''
        Remove points in pcd_sources that are close to the points in pcd_target
        '''
        
        cleaned_pcds = []
        for pcd in pcd_sources:
            pcd_cleaned = self.clean_around_label(pcd_target, pcd, tolerance)
            cleaned_pcds.append(pcd_cleaned)
        return cleaned_pcds
    
    def tilt_rectification(self, pcd_input: o3d.t.geometry.PointCloud) -> o3d.t.geometry.PointCloud:
        '''
        Tilt rectification for the input pointcloud
        '''
        R = self.compute_tilt_matrix(pcd_input)
        self.R = R
        # yaw, pitch, roll = RotationUtils.rotation_matrix_to_ypr(R)

        # self.logger.error(f"=================================")      
        # self.logger.error(f"yaw: {yaw:.2f} degrees, pitch: {pitch:.2f} degrees, roll: {roll:.2f} degrees")
        # self.logger.error(f"=================================\n")

        # R_ = RotationUtils.ypr_to_rotation_matrix(yaw, pitch, roll)
        # ax_x_, ax_y_, ax_z_ = RotationUtils.rotation_matrix_to_axis_angles(R_)

        # self.logger.info(f"=================================")      
        # self.logger.info(f"ax_x_: {ax_x_:.2f} degrees, ax_y_: {ax_y_:.2f} degrees, ax_z_: {ax_z_:.2f} degrees")
        # self.logger.info(f"=================================\n")

        # [before tilt rectification]
        # angle made by the normal vector with the [x-axis, y-axis, z-axis]
        axis_x, axis_y, axis_z = RotationUtils.rotation_matrix_to_axis_angles(R)
        
        self.logger.info(f"=================================")      
        self.logger.info(f"BEFORE TILT RECTIFICATION...")
        self.logger.info(f"Ground plane normal makes [{axis_x:.2f},{axis_y:.2f},{axis_z:.2f}] degrees"
                        " with the [x-axis, y-axis, z-axis] respectively!")
        self.logger.info(f"=================================\n")

        # exit(1)

        # sanity check
        normal, _ = self.get_class_plane(pcd_input, self.LABELS["NAVIGABLE_SPACE"]["id"])
        
        # self.logger.info(f"=================================")      
        # self.logger.info(f"[BEFORE] normal.shape: {normal.shape}")
        # self.logger.info(f"=================================\n")
        
        normal_ = np.dot(normal, R.T)
        
        # self.logger.warning(f"=================================")      
        # self.logger.warning(f"[AFTER] normal_.shape: {normal_.shape}")
        # self.logger.warning(f"=================================\n")

        # [after tilt rectification]
        # angle made by the normal vector with the [x-axis, y-axis, z-axis]
        angles = self.axis_angles(normal_)
        
        self.logger.warning(f"=================================")    
        self.logger.warning(f"AFTER TILT RECTIFICATION...")
        self.logger.warning(f"Ground plane normal makes [{angles[0]:.2f},{angles[1]:.2f},{angles[2]:.2f}] degrees"
                            " with the [x-axis, y-axis, z-axis] respectively!")
        self.logger.warning(f"=================================\n")
    
        # # angle between normal and y-axis should be close to 0 / 180 degrees
        # assert np.isclose(angles[1], 0, atol=0.01) or np.isclose(angles[1], 180, atol=0.1), \
        #     f"Error: angles_transformed[1] is {angles[1]}"\
        #     "but it should be close to 0 or 180 degrees (±0.1°)."\
        #     "Please check the tilt correction!"


        pcd_corrected = pcd_input.clone()
        
        # making y-axis perpendicular to the ground plane + right-handed coordinate system
        pcd_corrected.rotate(R, center=(0, 0, 0))
        return pcd_corrected
    
    def project_to_ground_plane(
        self, 
        pcd_navigable: o3d.t.geometry.PointCloud, 
        additional_pointclouds: List[o3d.t.geometry.PointCloud]
    ) -> List[o3d.t.geometry.PointCloud]:
        ''' Project the additional pointclouds to the navigable plane '''
        
        normal, inliers = self.get_class_plane(pcd_navigable, self.LABELS["NAVIGABLE_SPACE"]["id"])
        normal = normal / np.linalg.norm(normal)
        inliers_navigable = pcd_navigable.select_by_index(inliers)

        # compute angle with y-axis
        angle_y = self.axis_angles(normal)[1]
        # self.logger.info(f"Angle between normal and y-axis: {angle_y:.2f} degrees")

        # align normal with +y-axis 
        if angle_y < 0:
            normal = -normal

        mean_Y = float(np.mean(inliers_navigable.point['positions'].numpy()[:, 1]))
        self.logger.info(f"=================================")    
        self.logger.info(f"[BEFORE SHIFTING] Mean value of Y coordinates: {mean_Y}")
        self.logger.info(f"=================================\n")

        # shift pcd_navigable so mean_y_value becomes 0
        shift_vector = np.array([0, -mean_Y, 0], dtype=np.float32)
        inliers_navigable.point['positions'] = inliers_navigable.point['positions'] + shift_vector
        
        mean_Y = float(np.mean(inliers_navigable.point['positions'].numpy()[:, 1]))
        self.logger.info(f"=================================")    
        self.logger.info(f"[AFTER SHIFTING] Mean value of Y coordinates: {mean_Y}")
        self.logger.info(f"=================================\n")

        # Verify mean_Y is close to zero after shifting
        assert np.isclose(mean_Y, 0, atol=1e-6), f"Error: mean_Y ({mean_Y}) is not close to zero after shifting!"

        # label-wise BEV generations
        bev_pointclouds = []
        for pcd in (pcd_navigable, *additional_pointclouds):
            bev_pcd = pcd.clone()
            bev_pcd.point['positions'][:, 1] = float(mean_Y)
            bev_pointclouds.append(bev_pcd)

        return bev_pointclouds

    def clean_around_features(
        self, 
        bev_obstacle: o3d.t.geometry.PointCloud, 
        bev_pole: o3d.t.geometry.PointCloud, 
        bev_stem: o3d.t.geometry.PointCloud, 
        bev_canopy: o3d.t.geometry.PointCloud, 
        bev_navigable: o3d.t.geometry.PointCloud
    ) -> List[o3d.t.geometry.PointCloud]:
        
        '''Clean around each label in the BEV pointclouds'''

        # cleaning around obstacle
        # self.logger.warning(f"Cleaning around OBSTACLE!")
        [bev_navigable, bev_canopy, bev_stem, bev_pole] = self.clean_around_labels(bev_obstacle, [bev_navigable, bev_canopy, bev_stem, bev_pole], tolerance=0.02)
        
        # self.logger.info(f"[AFTER] len(bev_stem): {len(bev_stem.point['positions'])}")
        
        # cleaning around poles
        # self.logger.warning(f"Cleaning around POLE!")
        [bev_navigable, bev_canopy] = self.clean_around_labels(bev_pole, [bev_navigable, bev_canopy], tolerance=0.02)
        
        # cleaning around stem
        # self.logger.warning(f"Cleaning around STEM!")
        [bev_navigable, bev_canopy] = self.clean_around_labels(bev_stem, [bev_navigable, bev_canopy], tolerance=0.02)

        # cleaning around canopy
        # self.logger.warning(f"Cleaning around CANOPY!")
        [bev_navigable] = self.clean_around_labels(bev_canopy, [bev_navigable], tolerance=0.01)

        return [bev_navigable, bev_canopy, bev_stem, bev_pole, bev_obstacle]

    def generate_BEV(self, pcd_input: o3d.t.geometry.PointCloud) -> o3d.t.geometry.PointCloud:
        """Generate BEV from segmented pointcloud"""
        
        pcd_corrected = self.tilt_rectification(pcd_input)

        # filtering unwanted labels => [vegetation, tractor-hood, void, sky]
        valid_labels = np.array([label["id"] for label in self.LABELS.values()])
        valid_mask = np.isin(pcd_corrected.point['label'].numpy(), valid_labels)
        
        pcd_filtered = pcd_corrected.select_by_mask(valid_mask.flatten())
        original_points = len(pcd_corrected.point['positions'])
        filtered_points = len(pcd_filtered.point['positions'])
        reduction_percentage = ((original_points - filtered_points) / original_points) * 100    
        
        unique_labels = np.unique(pcd_filtered.point['label'].numpy())

        self.logger.info(f"=================================")    
        self.logger.info(f"Before filtering: {original_points}")
        self.logger.info(f"After filtering: {filtered_points}")
        self.logger.info(f"Reduction %: {reduction_percentage:.2f}%")
        self.logger.info(f"Unique labels in pcd_filtered: {unique_labels}")
        self.logger.info(f"=================================\n")
        
        # class-wise point cloud extraction
        pcd_canopy = self.get_class_pointcloud(pcd_filtered, self.LABELS["VINE_CANOPY"]["id"])
        pcd_pole = self.get_class_pointcloud(pcd_filtered, self.LABELS["VINE_POLE"]["id"])
        pcd_stem = self.get_class_pointcloud(pcd_filtered, self.LABELS["VINE_STEM"]["id"])
        pcd_obstacle = self.get_class_pointcloud(pcd_filtered, self.LABELS["OBSTACLE"]["id"])
        pcd_navigable = self.get_class_pointcloud(pcd_filtered, self.LABELS["NAVIGABLE_SPACE"]["id"])
        
        # num-points for each class
        total_points = len(pcd_filtered.point['positions'])
        canopy_points = len(pcd_canopy.point['positions'])
        pole_points = len(pcd_pole.point['positions'])
        stem_points = len(pcd_stem.point['positions'])
        obstacle_points = len(pcd_obstacle.point['positions'])
        navigable_points = len(pcd_navigable.point['positions'])
        
        # % points for each class
        canopy_percentage = (canopy_points / total_points) * 100
        pole_percentage = (pole_points / total_points) * 100
        stem_percentage = (stem_points / total_points) * 100
        obstacle_percentage = (obstacle_points / total_points) * 100
        navigable_percentage = (navigable_points / total_points) * 100

        self.logger.info(f"=================================")    
        self.logger.info(f"Total points: {total_points}")
        self.logger.info(f"Canopy points: {canopy_points} [{canopy_percentage:.2f}%]")
        self.logger.info(f"Pole points: {pole_points} [{pole_percentage:.2f}%]")
        self.logger.info(f"Stem points: {stem_points} [{stem_percentage:.2f}%]")
        self.logger.info(f"Obstacle points: {obstacle_points} [{obstacle_percentage:.2f}%]")
        self.logger.info(f"Navigable points: {navigable_points} [{navigable_percentage:.2f}%]")
        self.logger.info(f"=================================\n")

        # downsampling label-wise pointcloud
        down_pcd = pcd_filtered.voxel_down_sample(voxel_size=0.01)
        down_canopy = pcd_canopy.voxel_down_sample(voxel_size=0.01)
        down_navigable = pcd_navigable.voxel_down_sample(voxel_size=0.01)
          
        # NOT DOWN-SAMPLING [obstacle, stem, pole]
        down_obstacle = pcd_obstacle.clone()
        down_stem = pcd_stem.clone()
        down_pole = pcd_pole.clone()

        down_total_points = len(down_pcd.point['positions'].numpy())
        down_canopy_points = len(down_canopy.point['positions'])
        down_pole_points = len(down_pole.point['positions'])
        down_stem_points = len(down_stem.point['positions'])
        down_obstacle_points = len(down_obstacle.point['positions'])
        down_navigable_points = len(down_navigable.point['positions'])

        total_reduction_pct = (total_points - down_total_points) / total_points * 100 if total_points != 0 else 0
        canopy_reduction_pct = (canopy_points - down_canopy_points) / canopy_points * 100 if canopy_points != 0 else 0
        pole_reduction_pct = (pole_points - down_pole_points) / pole_points * 100 if pole_points != 0 else 0
        stem_reduction_pct = (stem_points - down_stem_points) / stem_points * 100 if stem_points != 0 else 0
        obstacle_reduction_pct = (obstacle_points - down_obstacle_points) / obstacle_points * 100 if obstacle_points != 0 else 0
        navigable_reduction_pct = (navigable_points - down_navigable_points) / navigable_points * 100 if navigable_points != 0 else 0
        
        # self.logger.info(f"=================================")    
        # self.logger.info(f"[AFTER DOWNSAMPLING]")
        # self.logger.info(f"Total points: {down_total_points} [-{total_reduction_pct:.2f}%]")
        # self.logger.info(f"Canopy points: {down_canopy_points} [-{canopy_reduction_pct:.2f}%]")
        # self.logger.info(f"Pole points: {down_pole_points} [-{pole_reduction_pct:.2f}%]")
        # self.logger.info(f"Stem points: {down_stem_points} [-{stem_reduction_pct:.2f}%]")
        # self.logger.info(f"Obstacle points: {down_obstacle_points} [-{obstacle_reduction_pct:.2f}%]")
        # self.logger.info(f"Navigable points: {down_navigable_points} [-{navigable_reduction_pct:.2f}%]")
        # self.logger.info(f"=================================\n")
        
        # radius-based outlier removal
        rad_filt_pole = down_pole if len(down_pole.point['positions']) == 0 else self.filter_radius_outliers(down_pole, nb_points=16, search_radius=0.05)[0]
        rad_filt_stem = down_stem if len(down_stem.point['positions']) == 0 else self.filter_radius_outliers(down_stem, nb_points=16, search_radius=0.05)[0]
        rad_filt_obstacle = down_obstacle if len(down_obstacle.point['positions']) == 0 else self.filter_radius_outliers(down_obstacle, nb_points=10, search_radius=0.05)[0]
        
        rad_filt_pole_points = len(rad_filt_pole.point['positions'].numpy())
        rad_filt_stem_points = len(rad_filt_stem.point['positions'].numpy())
        rad_filt_obstacle_points = len(rad_filt_obstacle.point['positions'].numpy())
        
        pole_reduction_pct = (down_pole_points - rad_filt_pole_points) / down_pole_points * 100 if down_pole_points != 0 else 0
        stem_reduction_pct = (down_stem_points - rad_filt_stem_points) / down_stem_points * 100 if down_stem_points != 0 else 0
        obstacle_reduction_pct = (down_obstacle_points - rad_filt_obstacle_points) / down_obstacle_points * 100 if down_obstacle_points != 0 else 0
        
        # self.logger.info(f"=================================")    
        # self.logger.info(f"[AFTER RADIUS-BASED OUTLIER REMOVAL]")
        # self.logger.info(f"Pole points: {rad_filt_pole_points} [-{pole_reduction_pct:.2f}%]")
        # self.logger.info(f"Stem points: {rad_filt_stem_points} [-{stem_reduction_pct:.2f}%]")
        # self.logger.info(f"Obstacle points: {rad_filt_obstacle_points} [-{obstacle_reduction_pct:.2f}%]")
        # self.logger.info(f"=================================\n")

        # projecting to ground plane
        bev_navigable, bev_canopy, bev_stem, bev_pole, bev_obstacle = (
            self.project_to_ground_plane(
                pcd_navigable, 
                [down_canopy, 
                rad_filt_stem, 
                rad_filt_pole, 
                rad_filt_obstacle]
            )
        )

        # bev_navigable, bev_canopy = (
        #     self.project_to_ground_plane(
        #         pcd_navigable, 
        #         [down_canopy] 
        #     )
        # )


        [bev_navigable, bev_canopy, bev_stem, bev_pole, bev_obstacle] = self.clean_around_features(
            bev_obstacle, 
            bev_pole, 
            bev_stem, 
            bev_canopy, 
            bev_navigable
        )

        bev_collection = [bev_navigable, bev_canopy, bev_stem, bev_pole, bev_obstacle]
        # bev_collection = [bev_navigable, bev_canopy]
        
        combined_pcd = self.generate_unified_bev_pcd(bev_collection)    

        unique_labels = np.unique(combined_pcd.point['label'].numpy())
        
        self.logger.info(f"=================================")    
        self.logger.info(f"Number of unique labels: {len(unique_labels)}")
        self.logger.info(f"Unique labels: {unique_labels}")
        self.logger.info(f"=================================\n")

        # debug_utils.plot_bev_scatter(bev_collection)

        return combined_pcd
    
    def bev_to_seg_mask_mono(self, pcd: o3d.t.geometry.PointCloud, 
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
        assert nx == nz, "nx and nz must be equal!"

        # Extract bounding box limits
        x_min, x_max = bb['x_min'], bb['x_max']
        z_min, z_max = bb['z_min'], bb['z_max']

        # Calculate grid resolution
        res_x = (x_max - x_min) / nx
        res_z = (z_max - z_min) / nz

        assert res_x == res_z, "Resolution x and z must be equal!"

        self.logger.info(f"=================================")    
        self.logger.info(f"Resolution X: {res_x:.2f} meters, Z: {res_z:.2f} meters")
        self.logger.info(f"=================================\n")

    
        # extract point coordinates and labels
        x_coords = pcd.point['positions'][:, 0].numpy()
        z_coords = pcd.point['positions'][:, 2].numpy()
        labels = pcd.point['label'].numpy()

        # generate mask_x and mask_z using res_x
        mask_x = ((x_coords - x_min) / res_x).astype(np.int32)
        assert mask_x.min() >= 0 and mask_x.max() < nx, "x-indices are out of bounds!"
        
        mask_z = nz - 1 - ((z_coords - z_min) / res_z).astype(np.int32)
        assert mask_z.min() >= 0 and mask_z.max() < nz, "z-indices are out of bounds!"
        
        # initialize mask
        mask = np.zeros((nz, nx), dtype=np.uint8)
        
        for x, z, label in zip(mask_x, mask_z, labels):
            mask[z, x] = label  

        return mask
  
    def pcd_to_seg_mask(self, 
                             pcd: o3d.t.geometry.PointCloud, 
                             nx: int = None, nz: int = None, 
                             bb: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        
        '''Generate mono / rgb segmentation masks from a pointcloud'''
                
        assert bb is not None, "Bounding box parameters are required!"
        assert nx is not None and nz is not None, "nx and nz must be provided!"
        
        bev_pcd = self.generate_BEV(pcd)
        bev_pcd_cropped = crop_pcd(bev_pcd, bb)

        seg_mask_mono = self.bev_to_seg_mask_mono(bev_pcd_cropped, nx, nz, bb)
        seg_mask_rgb = mono_to_rgb_mask(seg_mask_mono)

        return seg_mask_mono, seg_mask_rgb

    def get_updated_camera_extrinsics(self, pcd: o3d.t.geometry.PointCloud) -> np.ndarray:
        '''Get updated camera extrinsics after tilting the pointcloud'''
        
        assert self.R is not None, "Rotation matrix is required!"
        T = np.eye(4)
        R = self.R.T
        T[:3, :3] = R
        return T

  