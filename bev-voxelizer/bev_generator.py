#! /usr/bin/env python3
import logging, coloredlogs
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import open3d.core as o3c
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional
from logger import get_logger

import torch

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

        self.logger = get_logger("bev_generator", level=logging.ERROR)
    
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
        
        # pitch, yaw, roll
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
    
    def rotation_matrix_to_ypr(self,R):
        '''
        Convert rotation matrix to yaw, pitch, roll
        '''
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            yaw = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[1, 0], R[0, 0])
        else:
            yaw = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            roll = 0

        return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

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
            
        yaw, pitch, roll = self.rotation_matrix_to_ypr(R)

        # sanity check
        normal, _ = self.get_class_plane(pcd_input, self.LABELS["NAVIGABLE_SPACE"]["id"])
        
        self.logger.info(f"=================================")      
        self.logger.info(f"[BEFORE] normal.shape: {normal.shape}")
        self.logger.info(f"=================================\n")
        
        normal_ = np.dot(normal, R.T)
        
        self.logger.info(f"=================================")      
        self.logger.info(f"[AFTER] normal_.shape: {normal_.shape}")
        self.logger.info(f"=================================\n")

        # Calculate angles using the axis_angles function
        angles = self.axis_angles(normal_)
        
        # pitch: rotation around the x-axis.
        # roll: rotation around the z-axis
        # yaw: rotation around the y-axis
        pitch_, roll_, yaw_ = angles[0], angles[2], angles[1]

        self.logger.warning(f"=================================")    
        self.logger.warning(f"[BEFORE TILT RECTIFICATION] Yaw: {yaw:.2f} degrees, Pitch: {pitch:.2f} degrees, Roll: {roll:.2f} degrees")
        self.logger.warning(f"[AFTER  TILT RECTIFICATION] Yaw: {yaw_:.2f} degrees, Pitch: {pitch_:.2f} degrees, Roll: {roll_:.2f} degrees")
        self.logger.warning(f"=================================\n")

        # angle between normal and y-axis should be close to 0 / 180 degrees
        if not np.isclose(angles[1], 0, atol=1) and np.isclose(angles[1], 180, atol=1):
            self.logger.error(f"=================================")    
            self.logger.error(f"Error: angles_transformed[1] is {angles[1]}, but it should be close to 0 degrees. Please check the tilt correction!")
            self.logger.error(f"=================================\n")
            exit(1)

        pcd_corrected = pcd_input.clone()
        
        # making y-axis perpendicular to the ground plane + right-handed coordinate system
        pcd_corrected.rotate(R, center=(0, 0, 0))
        return pcd_corrected
    
    def project_to_ground_plane(
        self, 
        pcd_navigable: o3d.t.geometry.PointCloud, 
        additional_pointclouds: List[o3d.t.geometry.PointCloud]
    ) -> List[o3d.t.geometry.PointCloud]:
        ''' Project the input pointclouds to the navigable plane '''
        
        normal, inliers = self.get_class_plane(pcd_navigable, self.LABELS["NAVIGABLE_SPACE"]["id"])
        normal = normal / np.linalg.norm(normal)
        inliers_navigable = pcd_navigable.select_by_index(inliers)

        # compute angle with y-axis
        angle_y = self.axis_angles(normal)[1]
        # self.logger.info(f"Angle between normal and y-axis: {angle_y:.2f} degrees")

        # align normal with +y-axis if angle with y-axis is negative
        if angle_y < 0:
            normal = -normal

        mean_Y = float(np.mean(inliers_navigable.point['positions'].numpy()[:, 1]))
        self.logger.warning(f"=================================")    
        self.logger.warning(f"[BEFORE SHIFTING] Mean value of Y coordinates: {mean_Y}")
        self.logger.warning(f"=================================\n")

        # shift pcd_navigable so mean_y_value becomes 0
        shift_vector = np.array([0, -mean_Y, 0], dtype=np.float32)
        inliers_navigable.point['positions'] = inliers_navigable.point['positions'] + shift_vector
        
        mean_Y = float(np.mean(inliers_navigable.point['positions'].numpy()[:, 1]))
        self.logger.warning(f"=================================")    
        self.logger.warning(f"[AFTER SHIFTING] Mean value of Y coordinates: {mean_Y}")
        self.logger.warning(f"=================================\n")

        # Verify mean_Y is close to zero after shifting
        if not np.isclose(mean_Y, 0, atol=1e-6):
            self.logger.error(f"=================================")
            self.logger.error(f"Error: mean_Y ({mean_Y}) is not close to zero after shifting!")
            self.logger.error(f"=================================\n")
            exit(1)

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

    def pcd_to_segmentation_mask_mono(pcd: o3d.t.geometry.PointCloud, H: int = 480, W: int = 640, bb: dict = None) -> np.ndarray:
        """2D single-channel segmentation mask from BEV pcd."""
        # z-axis --> points into the scene is +ve
        # y-axis --> pointing down is +ve
        # x-axis --> pointing right is +ve

        assert bb is not None, "Bounding box parameters are required!"
        
        # extract point coordinates and labels
        x_coords = pcd.point['positions'][:, 0].numpy()
        z_coords = pcd.point['positions'][:, 2].numpy()
        labels = pcd.point['label'].numpy()

        x_min, x_max = bb['x_min'], bb['x_max']
        z_min, z_max = bb['z_min'], bb['z_max']

        # x_scaled = ((x_coords - x_min) / (x_max - x_min) * (W - 1)).astype(np.int32)
        # z_scaled = ((z_coords - z_min) / (z_max - z_min) * (H - 1)).astype(np.int32)

        # create empty mask
        mask = np.zeros((H, W), dtype=np.uint8)

        # label mapping (using original label values directly)
        # 1: Obstacle
        # 2: Navigable Space
        # 3: Vine Canopy  
        # 4: Vine Stem
        # 5: Vine Pole

        # # fill mask with label values
        # for x, z, label in zip(x_scaled, z_scaled, labels):
        #     if 1 <= label <= 5:  # only use valid label values
        #         mask[z, x] = label

        # fill mask with label values
        for x, z, label in zip(x_scaled, z_scaled, labels):
            mask[H - z - 1, x] = label  # invert z to match image coordinates

        return mask

if __name__ == "__main__":
    
    pcd_input = o3d.t.io.read_point_cloud("debug/left-segmented-labelled.ply")
    
    bev_generator = BEVGenerator()
    bev_pcd = bev_generator.generate_BEV(pcd_input)
    