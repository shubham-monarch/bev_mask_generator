#! /usr/bin/env python3

import boto3
import os
import random
from logger import get_logger
from typing import List
from tqdm import tqdm
import open3d as o3d
import numpy as np
import cv2

from bev_generator import BEVGenerator

class LeafFolder:
    def __init__(self, src_URI: str, dest_URI: str):
        '''
        :param src_URI: occ-dataset S3 URI
        :param dest_URI: bev-dataset S3 URI
        '''
        self.logger = get_logger("LeafFolder")
        self.src_URI = src_URI
        self.dest_URI = dest_URI
        self.s3 = boto3.client('s3')

        
        
        self.bev_generator = BEVGenerator()
    
    def process_folder(self):
        
        # ==================
        # 1. download left-segmented-labelled.ply
        # ==================
        pcd_URI = os.path.join(self.src_URI, "left-segmented-labelled.ply")

        self.logger.warning(f"=======================")
        self.logger.warning(f"pcd_URI: {pcd_URI}")
        self.logger.warning(f"=======================\n")

        pcd_path = self.download_file(pcd_URI)

        # ==================
        # 2. generate mono / RGB segmentation masks
        # ==================
        pcd = o3d.t.io.read_point_cloud(pcd_path)
        
        # mask dimensions
        nx, nz = 256, 256

        # z is depth, x is horizontal
        crop_bb = {'x_min': -2.5, 'x_max': 2.5, 'z_min': 0.0, 'z_max': 5}        
        
        seg_mask_mono, seg_mask_rgb = self.bev_generator.pcd_to_seg_mask(pcd,
                                                                         nx=256,nz=256,
                                                                         bb=crop_bb)
        
        # ==================
        # 3. upload mono / RGB segmentation masks
        # ==================
        self.upload_seg_mask(seg_mask_mono, self.dest_URI + "seg-mask-mono.png")
        self.upload_seg_mask(seg_mask_rgb, self.dest_URI + "seg-mask-rgb.png")
        
        # ==================
        # 4. process left / right images
        # ==================
        # download 1920x1080 
        imgL_uri = self.src_URI + "left.jpg"

        # resize  to 640 * 480
        # upload resized image 

        pass

        
    def rescale_img(self, img_uri: str):
        pass

    def copy_imgL(self, imgL_uri: str):
        pass

    def copy_imgR(self, imgR_uri: str):
        pass

    def upload_ipm_fea(self, ipm_fea_uri: str):
        pass

    def upload_ipm_rgb(self, ipm_seg_uri: str):
        pass

    def upload_file(self, src_path: str, dest_URI: str) -> bool:
        ''' Upload a file from src_path to dest_URI'''      
        
        try:
            bucket_name, key = dest_URI.replace("s3://", "").split("/", 1)
            self.s3.upload_file(src_path, bucket_name, key)
            return True
        except Exception as e:
            self.logger.error(f"Failed to upload file {src_path} to {dest_URI}: {str(e)}")
            return False

    def download_file(self, file_URI: str) -> str:
        ''' Download file from S3 to tmp-files folder'''
        
        try:
            bucket_name, key = file_URI.replace("s3://", "").split("/", 1)
            file_name = key.split("/")[-1]
            
            self.logger.info(f"=======================")
            self.logger.info(f"bucket_name: {bucket_name}")
            self.logger.info(f"key: {key}")
            self.logger.info(f"file_name: {file_name}")
            self.logger.info(f"=======================\n")

            os.makedirs("tmp-files", exist_ok=True)
            tmp_path = os.path.join("tmp-files", file_name)
            
            self.s3.download_file(bucket_name, key, tmp_path)
            return tmp_path
        except Exception as e:
            self.logger.error(f"Failed to download file from {file_URI}: {str(e)}")
            raise
    
    
    # def download_file(self, src_URI: str, dest_folder: str) -> str:
    #     ''' Download a file from S3 to the dest_folder'''
        
    #     try:
    #         bucket_name, key = src_URI.replace("s3://", "").split("/", 1)
    #         file_name = key.split("/")[-1]
            
    #         os.makedirs(dest_folder, exist_ok=True)
    #         tmp_path = os.path.join(dest_folder, file_name)
            
    #         self.s3.download_file(bucket_name, key, tmp_path)
    #         return tmp_path
    #     except Exception as e:
    #         self.logger.error(f"Failed to download file from {src_URI} to {dest_folder}: {str(e)}")
    #         raise
    
    
    def upload_seg_mask(self, mask: np.ndarray, mask_uri: str) -> bool:
        """Save mask as PNG and upload to S3"""
        
        os.makedirs("tmp-masks", exist_ok=True)
        tmp_path = os.path.join("tmp-masks", "tmp_mask.png")
        cv2.imwrite(tmp_path, mask)
        
        # upload to S3
        success = self.upload_file(tmp_path, mask_uri)
        
        # clean-up   
        if success:
            os.remove(tmp_path)
        
        return success


    # def download_segmented_pcd(self, folder_URI: str) -> str:
    #     self.logger.info(f"=======================")
    #     self.logger.info(f"Downloading left-segmented-labelled.ply!")
    #     self.logger.info(f"=======================\n")
        
    #     segmented_pcd_uri = folder_URI + f"left-segmented-labelled.ply" 
    #     return self.download_file(segmented_pcd_uri, "tmp-pcd")
                

        

    def generate_bev(self, imgL_uri: str, imgR_uri: str):
        pass
    

class DataGeneratorS3:
    def __init__(self, src_URIs: List[str] = None, dest_folder: str = None, index_json: str = None):    
        self.logger = get_logger("DataGeneratorS3")
        self.src_URIs = src_URIs

        
    def get_target_folder_uri(self, src_uri: str, dest_folder:str = "bev-dataset"):
        ''' Make leaf-folder path relative to the bev-dataset folder '''
        return src_uri.replace("occ-dataset", dest_folder, 1)
        
   
    def get_leaf_folders(self) -> List[str]:
        """Get all leaf folders URI inside the given S3 URIs"""
        
        all_leaf_uris = []
        
        for s3_uri in self.src_URIs:
            # Parse S3 URI to get bucket and prefix
            s3_parts = s3_uri.replace("s3://", "").split("/", 1)
            bucket_name = s3_parts[0]
            prefix = s3_parts[1] if len(s3_parts) > 1 else ""
            
            # Initialize S3 client
            s3 = boto3.client('s3')
            
            # Get all objects with the given prefix
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
            
            # Keep track of all folders and their parent folders
            all_folders = set()
            parent_folders = set()
            
            # Process all objects
            for page in pages:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    # Get the full path
                    path = obj['Key']
                    
                    # Skip the prefix itself
                    if path == prefix:
                        continue
                        
                    # Get all folder paths in this object's path
                    parts = path.split('/')
                    for i in range(len(parts)-1):
                        folder = '/'.join(parts[:i+1])
                        if folder:
                            all_folders.add(folder)
                            
                            # If this isn't the immediate parent, it's a parent folder
                            if i < len(parts)-2:
                                parent_folders.add(folder)
            
            # Leaf folders are those that aren't parents of other folders
            leaf_folders = all_folders - parent_folders
            
            # Convert back to S3 URIs and add to list
            leaf_folder_uris = [f"s3://{bucket_name}/{folder}" for folder in sorted(leaf_folders)]
            all_leaf_uris.extend(leaf_folder_uris)
        
        return all_leaf_uris


    def generate_bev_dataset(self, src_uri: str, dest_folder: str = "bev-dataset"):
        ''' Generate a BEV dataset from the given S3 URI '''
        
        leaf_URIs = self.get_leaf_folders()
        random.shuffle(leaf_URIs)
        
        for idx, leaf_URI in tqdm(enumerate(leaf_URIs), total=len(leaf_URIs), desc="Processing leaf URIs"):    
            target_folder = self.get_target_folder_uri(leaf_URI, dest_folder)
            leaf_folder = LeafFolder(leaf_URI, target_folder)
            leaf_folder.process_folder()


if __name__ == "__main__":
    URIs = [
        "s3://occupancy-dataset/occ-dataset/vineyards/gallo/",
        "s3://occupancy-dataset/occ-dataset/vineyards/RJM/"
    ]

    logger = get_logger("__main__")
    
    data_generator_s3 = DataGeneratorS3(src_URIs=URIs)
    leaf_URIs = data_generator_s3.get_leaf_folders()

    leaf_URI_src = random.choice(leaf_URIs)
    leaf_URI_dest = data_generator_s3.get_target_folder_uri(leaf_URI_src)

    logger.warning(f"=======================")
    logger.warning(f"leaf_URI_src: {leaf_URI_src}")
    logger.warning(f"leaf_URI_dest: {leaf_URI_dest}")
    logger.warning(f"=======================\n")

    # src_URI = "s3://occupancy-dataset/occ-dataset/vineyards/RJM/"
    
    # leaf_URI_src = \
    #     "s3://occupancy-dataset/" \
    #     "occ-dataset/" \
    #     "vineyards/gallo/" \
    #     "2024_06_07_utc/svo_files/front_2024-06-04-10-24-57.svo/1398_to_1540/frame-1400/"

    # leaf_URI_dest = \
    #     "s3://occupancy-dataset/" \
    #     "bev-dataset/" \
    #     "vineyards/gallo/" \
    #     "2024_06_07_utc/svo_files/front_2024-06-04-10-24-57.svo/1398_to_1540/frame-1400/"
    
    # src_URI = leaf_URI

    # data_generator_s3.generate_bev_dataset(leaf_URI_src)
    
    # logger.warning(f"=======================")
    # logger.warning(f"leaf_URI_src: {leaf_URI_src}")
    # logger.warning(f"leaf_URI_dest: {leaf_URI_dest}")
    # logger.warning(f"=======================\n")

    leafFolder = LeafFolder(leaf_URI_src, leaf_URI_dest)
    leafFolder.process_folder()
