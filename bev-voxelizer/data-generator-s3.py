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
import json
from bev_generator import BEVGenerator


class JSONIndex:
    def __init__(self, json_path: str = None):
        
        assert json_path is not None, "json_path is required!"
        
        self.index_path = json_path
        self.index = self.load_index(json_path)
        self.keys = ['seg-mask-mono', 'seg-mask-rgb', 'left-img', 'right-img']

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
        
        if file_uri not in self.index:
            self.index[file_uri] = {}
        
        # set all keys to false
        for key in self.keys:
            self.index[file_uri][key] = False


    def update_file(self, file_uri: str, key: str):
        ''' Update an existing file in the index '''
        
        assert key in self.keys, f"Key {key} not found in index"
        assert file_uri in self.index, f"File {file_uri} not found in index"

        self.index[file_uri][key] = True
        

    def save_index(self):
        '''Save index as json to disk'''
        
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=4)

    def check_index(self, file_uri: str, key: str) -> bool:
        ''' Check if the index has the given file and key '''
        
        if file_uri not in self.index:
            return False
        
        if key not in self.index[file_uri]:
            return False
        
        return self.index[file_uri][key]
    

class LeafFolder:
    def __init__(self, src_URI: str, dest_URI: str, index_json: str = None):
        '''
        :param src_URI: occ-dataset S3 URI
        :param dest_URI: bev-dataset S3 URI
        :param index_json: index.json file path
        '''
        assert index_json is not None, "index_json is required!"

        self.logger = get_logger("LeafFolder")
        
        self.src_URI = src_URI
        self.dest_URI = dest_URI
        
        self.logger.warning(f"=======================")
        self.logger.warning(f"src_URI: {self.src_URI}")
        self.logger.warning(f"dest_URI: {self.dest_URI}")
        self.logger.warning(f"=======================\n")

        self.s3 = boto3.client('s3')    
        self.tmp_folder = "tmp-files"
        self.index = JSONIndex(index_json)
        self.bev_generator = BEVGenerator()
        

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
            
            # self.logger.info(f"=======================")
            # self.logger.info(f"bucket_name: {bucket_name}")
            # self.logger.info(f"key: {key}")
            # self.logger.info(f"file_name: {file_name}")
            # self.logger.info(f"=======================\n")

            os.makedirs("tmp-files", exist_ok=True)
            tmp_path = os.path.join("tmp-files", file_name)
            
            self.s3.download_file(bucket_name, key, tmp_path)
            return tmp_path
        except Exception as e:
            self.logger.error(f"Failed to download file from {file_URI}: {str(e)}")
            raise    
    
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

    def process_folder(self):
        ''' Process a folder and generate BEV dataset '''
        
        self.index.add_file(self.src_URI)
        
        # ==================
        # 1. download left-segmented-labelled.ply
        # ==================

        self.logger.info(f"=======================")
        self.logger.info(f"[STEP #1]: downloading left-segmented-labelled.ply...")
        self.logger.info(f"=======================\n")

        pcd_URI = os.path.join(self.src_URI, "left-segmented-labelled.ply")

        # self.logger.info(f"=======================")
        # self.logger.info(f"pcd_URI: {pcd_URI}")
        # self.logger.info(f"=======================\n")

        pcd_path = self.download_file(pcd_URI)

        # ==================
        # 2. generate mono / RGB segmentation masks
        # ==================

        self.logger.info(f"=======================")
        self.logger.info(f"[STEP #2]: generating mono / RGB segmentation masks...")
        self.logger.info(f"=======================\n")

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
        
        self.logger.info(f"=======================")
        self.logger.info(f"[STEP #3]: uploading mono / RGB segmentation masks...")
        self.logger.info(f"=======================\n")

        self.upload_seg_mask(seg_mask_mono, os.path.join(self.dest_URI, "seg-mask-mono.png"))
        self.upload_seg_mask(seg_mask_rgb, os.path.join(self.dest_URI, "seg-mask-rgb.png"))
        
        # update index
        self.index.update_file(self.src_URI, 'seg-mask-mono')
        self.index.update_file(self.src_URI, 'seg-mask-rgb')

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

        # update index
        self.index.update_file(self.src_URI, 'left-img')
        self.index.update_file(self.src_URI, 'right-img')

        # save index
        self.index.save_index()


class DataGeneratorS3:
    def __init__(self, src_URIs: List[str] = None, dest_folder: str = None, index_json: str = None):    
        
        assert index_json is not None, "index_json is required!"

        self.logger = get_logger("DataGeneratorS3")
        self.src_URIs = src_URIs
        self.index_json = index_json
        
    def generate_target_URI(self, src_uri: str, dest_folder:str = None):
        ''' Make leaf-folder path relative to the bev-dataset folder '''
        
        assert dest_folder is not None, "dest_folder is required!"
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


    def generate_bev_dataset(self, dest_folder: str = "bev-dataset"):
        ''' Generate a BEV dataset from the given S3 URI '''
        
        assert dest_folder == "bev-dataset", "dest_folder must be 'bev-dataset'"

        self.logger.info(f"=======================")
        self.logger.info(f"STARTING BEV-S3-DATASET GENERATION PIPELINE...")
        self.logger.info(f"=======================\n")

        leaf_URIs = self.get_leaf_folders()
        random.shuffle(leaf_URIs)
        
        for idx, src_URI in tqdm(enumerate(leaf_URIs), total=len(leaf_URIs), desc="Processing leaf URIs"):    
            target_URI = self.generate_target_URI(src_URI, dest_folder)
            leaf_folder = LeafFolder(src_URI, target_URI, self.index_json)
            leaf_folder.process_folder()


if __name__ == "__main__":
    URIs = [
        "s3://occupancy-dataset/occ-dataset/vineyards/gallo/",
        "s3://occupancy-dataset/occ-dataset/vineyards/RJM/"
    ]

    logger = get_logger("__main__")
    
    json_path = "index.json"
    data_generator_s3 = DataGeneratorS3(src_URIs=URIs, index_json=json_path)
    data_generator_s3.generate_bev_dataset()
    