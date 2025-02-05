#! /usr/bin/env python3

import boto3
import os
import random
from typing import List
from tqdm import tqdm
import open3d as o3d
import numpy as np
import cv2
import json
import logging
import yaml
import argparse

from scripts.logger import get_logger
# from scripts.vineyards_mask_generator import BEVGenerator
from scripts.dairy_mask_generator import BEVGenerator


class JSONIndex:
    def __init__(self, json_path: str = None):
        
        assert json_path is not None, "json_path is required!"
        
        self.index_path = json_path
        self.index = self.load_index(self.index_path)
        self.keys = ['seg-mask-mono', 'seg-mask-rgb', 'left-img', 'right-img', 'cam-extrinsics']

    def load_index(self, json_path: str) -> dict:
        ''' Load index.json file from disk '''

        if json_path and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {}

        return self.index
    
    def add_file(self, file_uri: str):
        ''' Add a new file to the index '''
        
        if not self.has_file(file_uri):
            self.index[file_uri] = {}

    def save_index(self):
        '''Save index as json to disk'''
        
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=4)

    def has_file(self, file_uri: str) -> bool:
        ''' Check if the index has the given file '''
        
        return file_uri in self.index


class LeafFolder:
    def __init__(self, src_URI: str, dest_URI: str,
                 index_json: str = None,
                 crop_bb: dict = None,
                 color_map: str = None,
                 nx: int = None,
                 nz: int = None):
        '''
        :param src_URI: occ-dataset S3 URI
        :param dest_URI: bev-dataset S3 URI
        :param index_json: index.json file path
        '''
        assert index_json is not None, "index_json is required!"
        assert crop_bb is not None, "crop_bb is required!"
        assert color_map is not None, "color_map is required!"
        assert nx is not None, "nx is required!"
        assert nz is not None, "nz is required!"
        
        self.logger = get_logger("leaf-folder", level=logging.WARNING)
        
        self.src_URI = src_URI
        self.dest_URI = dest_URI
        
        # mavis.yaml
        self.color_map = color_map
        
        # crop bounding box
        self.crop_bb = crop_bb
        
        # segentation mask dimensions
        self.nx = nx
        self.nz = nz
        
        self.logger.info(f"=======================")
        self.logger.info(f"src_URI: {self.src_URI}")
        self.logger.info(f"dest_URI: {self.dest_URI}")
        self.logger.info(f"=======================\n")

        self.logger.info(f"=======================")
        self.logger.info(f"color_map: {self.color_map}")
        self.logger.info(f"crop_bb: {self.crop_bb}")
        self.logger.info(f"(nx, nz): ({self.nx}, {self.nz})")
        self.logger.info(f"=======================\n")

        self.s3 = boto3.client('s3')    
        self.tmp_folder = "tmp-files"
        self.INDEX = JSONIndex(index_json)
        self.bev_generator = BEVGenerator(yaml_path=self.color_map)
        

    def upload_file(self, src_path: str, dest_URI: str) -> bool:
        ''' Upload a file from src_path to dest_URI'''      
        
        try:
            bucket_name, key = dest_URI.replace("s3://", "").split("/", 1)
            self.s3.upload_file(src_path, bucket_name, key)
            return True
        except Exception as e:
            self.logger.error(f"Failed to upload file {src_path} to {dest_URI}: {str(e)}")
            return False

    @staticmethod
    def download_file(file_URI: str, tmp_folder: str = "tmp-files", log_level: int = logging.INFO) -> str:
        """Download file from S3 to tmp-files folder and return the local path
        
        Args:
            file_URI: S3 URI of the file to download
            tmp_folder: local folder to store downloaded files
            log_level: logging level to use (default: logging.INFO)
            
        Returns:
            str: path to the downloaded file
            
        Raises:
            Exception: if download fails
        """
        # Fixed: Changed 'level' to 'log_level' in get_logger call
        # logger = get_logger("download-file", level=log_level)

        # logger = get_logger("download-file", level = logging.ERROR)
        logger = get_logger("download-file", level = log_level)
          
          
        try:
            bucket_name, key = file_URI.replace("s3://", "").split("/", 1)
            file_name = key.split("/")[-1]
            
            os.makedirs(tmp_folder, exist_ok=True)
            tmp_path = os.path.join(tmp_folder, file_name)
            
            s3 = boto3.client('s3')
            logger.info(f"downloading {file_name}...")
            s3.download_file(bucket_name, key, tmp_path)
            logger.info(f"successfully downloaded {file_name}")
            
            return tmp_path
            
        except Exception as e:
            logger.error(f"failed to download {file_URI}: {str(e)}")
            raise

    def upload_seg_mask(self, mask: np.ndarray, mask_uri: str) -> bool:
        """Save mask as PNG and upload to S3"""
        
        os.makedirs(self.tmp_folder, exist_ok=True)
        tmp_path = os.path.join(self.tmp_folder, "tmp_mask.png")
        cv2.imwrite(tmp_path, mask)
        
        # upload to S3
        success = self.upload_file(tmp_path, mask_uri)
        
        # clean-up   
        if success:
            os.remove(tmp_path)
        
        return success

    def process_folder(self):
        ''' Process a folder and generate BEV dataset '''
        
        if self.INDEX.has_file(self.src_URI):
            self.logger.error(f"=======================")
            self.logger.error(f"Skipping folder {self.src_URI}...")
            self.logger.error(f"=======================\n")
            return

        self.logger.warning("───────────────────────────────")
        self.logger.warning(f"Processing folder {self.src_URI}")
        self.logger.warning("───────────────────────────────")

        # ==================
        # 1. download left-segmented-labelled.ply
        # ==================

        self.logger.info(f"=======================")
        self.logger.info(f"[STEP #1]: downloading left-segmented-labelled.ply...")
        self.logger.info(f"=======================\n")

        pcd_URI = os.path.join(self.src_URI, "left-segmented-labelled.ply")

        pcd_path = LeafFolder.download_file(pcd_URI, self.tmp_folder)

        # ==================
        # 2. generate mono / RGB segmentation masks
        # ==================

        self.logger.info(f"=======================")
        self.logger.info(f"[STEP #2]: generating mono / RGB segmentation masks...")
        self.logger.info(f"=======================\n")

        pcd = o3d.t.io.read_point_cloud(pcd_path)
        
        seg_mask_mono, seg_mask_rgb = self.bev_generator.pcd_to_seg_mask(pcd,
                                                                        nx=self.nx, nz=self.nz,
                                                                        bb=self.crop_bb)

        # ==================
        # 3. upload mono / RGB segmentation masks
        # ==================
    
        self.logger.info(f"=======================")
        self.logger.info(f"[STEP #3]: uploading mono / RGB segmentation masks...")
        self.logger.info(f"=======================\n")

        self.upload_seg_mask(seg_mask_mono, os.path.join(self.dest_URI, "seg-mask-mono.png"))
        self.upload_seg_mask(seg_mask_rgb, os.path.join(self.dest_URI, "seg-mask-rgb.png"))
        
            
        # ==================
        # 4. process left / right images
        # ==================
    
        self.logger.info(f"=======================")
        self.logger.info(f"[STEP #4]: resizing + uploading left / right image...")
        self.logger.info(f"=======================\n")

        # download 1920x1080 
        imgL_uri = os.path.join(self.src_URI, "left.jpg")
        imgL_path = self.download_file(imgL_uri)
        
        imgR_uri = os.path.join(self.src_URI, "right.jpg")
        imgR_path = self.download_file(imgR_uri)
        
        # resize to 640x480
        imgL = cv2.imread(imgL_path)
        imgR = cv2.imread(imgR_path)
        
        imgL_resized = cv2.resize(imgL, (640, 480))
        imgR_resized = cv2.resize(imgR, (640, 480))
        
        # save to tmp-folder
        imgL_path = os.path.join(self.tmp_folder, "left-resized.jpg")
        imgR_path = os.path.join(self.tmp_folder, "right-resized.jpg")
        
        cv2.imwrite(imgL_path, imgL_resized)
        cv2.imwrite(imgR_path, imgR_resized)
        
        # upload resized image 
        self.upload_file(imgL_path, os.path.join(self.dest_URI, "left.jpg"))
        self.upload_file(imgR_path, os.path.join(self.dest_URI, "right.jpg"))


        # =================
        # 5. upload camera extrinsics
        # =================

        self.logger.info(f"=======================")
        self.logger.info(f"[STEP #5]: uploading camera extrinsics...")
        self.logger.info(f"=======================\n")

        cam_extrinsics = self.bev_generator.get_updated_camera_extrinsics()
        
        # save to tmp-folder
        cam_extrinsics_path = os.path.join(self.tmp_folder, "cam-extrinsics.npy")
        np.save(cam_extrinsics_path, cam_extrinsics)
        
        # upload to S3
        self.upload_file(cam_extrinsics_path, os.path.join(self.dest_URI, "cam-extrinsics.npy"))

        # =================
        # 6. save index
        # =================
        self.INDEX.add_file(self.src_URI)
        self.INDEX.save_index()

        

class DataGeneratorS3:
    def __init__(self, config_path: str):
        """Initialize DataGeneratorS3 from a config file
        
        Args:
            config_path: Path to YAML config file containing parameters
        """
        self.logger = get_logger("data-generator-s3")
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate required parameters
        required_params = ['src_URIs', 'dest_folder', 'index_json', 
                         'color_map', 'crop_bb', 'nx', 'nz']
        for param in required_params:
            assert param in config, f"{param} is required in config!"
        
        # Set instance variables from config
        self.src_URIs = config['src_URIs']
        self.dest_folder = config['dest_folder']
        self.color_map = config['color_map']
        self.index_json = config['index_json']
        self.crop_bb = config['crop_bb']
        self.nx = config['nx']
        self.nz = config['nz']

    def generate_target_URI(self, src_uri: str, dest_folder:str = None):
        ''' Make leaf-folder path relative to the bev-dataset folder '''
        
        assert dest_folder is not None, "dest_folder is required!"
        return src_uri.replace("occ-dataset", dest_folder, 1)
        
    @staticmethod
    def get_leaf_folders(src_URIs: List[str]) -> List[str]:
        """Get all leaf folders URI inside the given S3 URIs
        
        Args:
            src_URIs: List of S3 URIs to search for leaf folders
            
        Returns:
            List of S3 URIs for all leaf folders found
        """
        all_leaf_uris = []
        
        for s3_uri in src_URIs:
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

    def generate_bev_dataset(self, dest_folder: str = None):
        ''' Generate a BEV dataset from the given S3 URI '''
            
        self.logger.info(f"=======================")
        self.logger.info(f"STARTING BEV-S3-DATASET GENERATION PIPELINE...")
        self.logger.info(f"=======================\n")

        # leaf_URIs = self.get_leaf_folders(self.src_URIs)
        leaf_URIs = DataGeneratorS3.get_leaf_folders(self.src_URIs)
        random.shuffle(leaf_URIs)
        
        for idx, src_URI in tqdm(enumerate(leaf_URIs), total=len(leaf_URIs), desc=f"Processing leaf URIs\n"):    
            target_URI = self.generate_target_URI(src_URI, self.dest_folder)
            
            leaf_folder = LeafFolder(src_URI, target_URI, 
                                     self.index_json, 
                                     self.crop_bb, 
                                     self.color_map, 
                                     self.nx, self.nz)
            try:
                leaf_folder.process_folder()
            except Exception as e:
                self.logger.error(f"Failed to process folder {src_URI}: {str(e)}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate BEV dataset from S3')
    parser.add_argument('--config_path', type=str, nargs='?',
                       help='Path to YAML config file')
    
    args = parser.parse_args()
    
    logger = get_logger("data-generator-s3")
    
    data_generator_s3 = DataGeneratorS3(config_path=args.config_path)
    data_generator_s3.generate_bev_dataset()
    
    # =================
    # counting processed frames
    # =================
    # BEV_URIs = [
    #     "s3://occupancy-dataset/bev-dataset-2-to-7/vineyards/gallo/",
    #     "s3://occupancy-dataset/bev-dataset-2-to-7/vineyards/RJM/"
    # ]
    # data_generator_s3 = DataGeneratorS3(src_URIs=BEV_URIs, 
    #                                     dest_folder="bev-dataset-2-to-7",
    #                                     index_json=json_path, 
    #                                     crop_bb=crop_bb)
    
    # leaf_folder_cnt = len(data_generator_s3.get_leaf_folders())
    
    # logger.info(f"=======================")
    # logger.info(f"Total leaf folders: {leaf_folder_cnt}")
    # logger.info(f"=======================\n")
    