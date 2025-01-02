#! /usr/bin/env python3

import numpy as np
import boto3
import os
import random
from tqdm import tqdm
import shutil
import open3d as o3d
import cv2

from logger import get_logger
from bev_generator import BEVGenerator


class DataGenerator:
    '''Class to generate training data for the SBEVNet model'''
    
    def __init__(self):
        self.logger = get_logger("data-generator")
    
    def rescale_images(self, src_folder = None, dst_folder = None, h = None, w = None):
        '''Rescale images from (1920, 1080) to (h, w)'''

        assert src_folder is not None, "src_folder is required"
        assert dst_folder is not None, "dst_folder is required"
        assert h is not None, "h is required"
        assert w is not None, "w is required"
        assert not (os.path.exists(dst_folder) and os.listdir(dst_folder)), "dst_folder must be empty"

        # Create destination folder if it doesn't exist
        os.makedirs(dst_folder, exist_ok=True)

        self.logger.info(f"=========================")
        self.logger.info(f"Rescaling images from {src_folder} to {dst_folder}")
        self.logger.info(f"=========================\n")

        # Copy all subfolders from src to dst
        for item in os.listdir(src_folder):
            src_path = os.path.join(src_folder, item)
            dst_path = os.path.join(dst_folder, item)
            
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

        # rescale all images in left and right folders
        for folder in ['left', 'right']:
            folder_path = os.path.join(dst_folder, folder)
            if not os.path.exists(folder_path):
                continue
                
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    resized_img = cv2.resize(img, (w, h))
                    cv2.imwrite(img_path, resized_img)
                except Exception as e:
                    self.logger.error(f"Failed to resize {img_path}: {str(e)}")


    def raw_data_to_model_data(self, raw_data_dir = None, model_data_dir = None):
        '''Process raw data and move to model-data folder'''

        assert raw_data_dir is not None, "raw_data_dir is required"
        assert model_data_dir is not None, "model_data_dir is required"        
        assert not (os.path.exists(model_data_dir) and os.listdir(model_data_dir)), "model_data_dir must be empty"

        self.logger.info(f"=========================")
        self.logger.info(f"Processing raw data from {raw_data_dir} to {model_data_dir}")
        self.logger.info(f"=========================\n")

        # Create the target folder if it doesn't exist
        os.makedirs(model_data_dir, exist_ok=True)

        # Define the target subfolders
        left_folder = os.path.join(model_data_dir, 'left')
        right_folder = os.path.join(model_data_dir, 'right')
        seg_masks_mono_folder = os.path.join(model_data_dir, 'seg-masks-mono')
        seg_masks_rgb_folder = os.path.join(model_data_dir, 'seg-masks-rgb')
        cam_extrinsics_folder = os.path.join(model_data_dir, 'cam-extrinsics')


        # Create the target subfolders if they don't exist
        os.makedirs(left_folder, exist_ok=True)
        os.makedirs(right_folder, exist_ok=True)
        os.makedirs(seg_masks_mono_folder, exist_ok=True)
        os.makedirs(seg_masks_rgb_folder, exist_ok=True)
        os.makedirs(cam_extrinsics_folder, exist_ok=True)

        # Count total files for progress bar
        total_files = 0
        for root, dirs, files in os.walk(raw_data_dir):
            for file in files:
                if file.endswith('left.jpg') or \
                   file.endswith('right.jpg') or \
                   file.endswith('-mono.png') or \
                   file.endswith('-rgb.png') or \
                   file.endswith('cam-extrinsics.npy'):
                    total_files += 1
        
        self.logger.info(f"=========================")
        self.logger.info(f"Total files: {total_files}") 
        self.logger.info(f"=========================\n")

        with tqdm(total=total_files, desc="Organizing Images") as pbar:
            for root, dirs, files in os.walk(raw_data_dir):
                # Get folder number from root path
                folder_num = os.path.basename(root)
                if not folder_num.isdigit():
                    continue

                for file in files:
                    if file == 'left.jpg':
                        new_filename = f"{folder_num}__left.jpg"
                        shutil.copy(os.path.join(root, file), os.path.join(left_folder, new_filename))
                        pbar.update(1)
                    elif file == 'right.jpg':
                        new_filename = f"{folder_num}__right.jpg"
                        shutil.copy(os.path.join(root, file), os.path.join(right_folder, new_filename))
                        pbar.update(1)
                    elif file == 'seg_mask_mono.png':
                        new_filename = f"{folder_num}__seg-mask-mono.png"
                        shutil.copy(os.path.join(root, file), os.path.join(seg_masks_mono_folder, new_filename))
                        pbar.update(1)
                    elif file == 'seg_mask_rgb.png':
                        new_filename = f"{folder_num}__seg-mask-rgb.png"
                        shutil.copy(os.path.join(root, file), os.path.join(seg_masks_rgb_folder, new_filename))
                        pbar.update(1)
                    elif file == 'cam_extrinsics.npy':
                        new_filename = f"{folder_num}__cam-extrinsics.npy"
                        shutil.copy(os.path.join(root, file), os.path.join(cam_extrinsics_folder, new_filename))
                        pbar.update(1)


    def s3_data_to_raw_data(self, s3_data_dir = None, raw_data_dir = None):
        '''Process s3 data and move to model-data folder'''

        assert s3_data_dir is not None, "s3_data_dir is required"
        assert raw_data_dir is not None, "raw_data_dir is required"    
        assert not (os.path.exists(raw_data_dir) and os.listdir(raw_data_dir)), "raw_data_dir must be empty"



        # walk through s3 data directory
        total_files = sum(1 for root, _, files in os.walk(s3_data_dir) 
                         for file in files if file == 'left-segmented-labelled.ply')
        
        with tqdm(total=total_files, desc="Processing files") as pbar:
            for root, dirs, files in os.walk(s3_data_dir):
                for file in files:
                    if file == 'left-segmented-labelled.ply':
                        try:
                            file_path = os.path.join(root, file)
                            pcd_input = o3d.t.io.read_point_cloud(file_path)
                        
                            bev_generator = BEVGenerator()
                        
                            # mask dimensions (400 * 400)
                            nx , ny = 400, 400
                            
                            # crop bounding box
                            crop_bb = {'x_min': -5, 'x_max': 5, 'z_min': 0, 'z_max': 10}

                            # mono / rgb segmentation masks
                            try:
                                seg_mask_mono, seg_mask_rgb = bev_generator.pcd_to_seg_mask(pcd_input,
                                                                                            nx, ny,
                                                                                            crop_bb)
                            except Exception as e:
                                self.logger.error(f"Failed to generate segmentation masks: {str(e)}")
                                raise
                            
                            # camera extrinsics
                            camera_extrinsics = bev_generator.get_updated_camera_extrinsics(pcd_input)

                            rel_path = os.path.relpath(root, s3_data_dir)
                            dest_dir = os.path.join(raw_data_dir, rel_path)
                            os.makedirs(dest_dir, exist_ok=True)
                            
                            # ================================================
                            # copy left and right images
                            # ================================================
                            for img_file in ['left.jpg', 'right.jpg']:
                                try:
                                    src_file = os.path.join(root, img_file)
                                    if os.path.exists(src_file):
                                        dest_file = os.path.join(dest_dir, img_file)
                                        shutil.copy2(src_file, dest_file)
                                except (IOError, OSError) as e:
                                    self.logger.error(f"Failed to copy {img_file}: {str(e)}")

                            # ================================================
                            # save segmentation masks
                            # ================================================
                            try:
                                cv2.imwrite(os.path.join(dest_dir, 'seg_mask_mono.png'), seg_mask_mono)
                                cv2.imwrite(os.path.join(dest_dir, 'seg_mask_rgb.png'), seg_mask_rgb)
                            except Exception as e:
                                self.logger.error(f"Failed to save segmentation masks: {str(e)}")

                            # ================================================
                            # save camera extrinsics
                            # ================================================
                            try:
                                np.save(os.path.join(dest_dir, 'cam_extrinsics.npy'), camera_extrinsics)
                            except Exception as e:
                                self.logger.error(f"Failed to save camera extrinsics: {str(e)}")
                            
                            pbar.update(1)
                            
                        except Exception as e:
                            self.logger.error(f"Failed to process {file}: {str(e)}")
                            continue


    def fetch_data_from_s3(self, s3_uri, local_dir, num_files=200):
        '''Fetch data from s3 and save to local directory'''
        
        assert not (os.path.exists(local_dir) and os.listdir(local_dir)), f"local_dir must be empty!"

        s3 = boto3.resource('s3')
        bucket_name, prefix = s3_uri.replace("s3://", "").split("/", 1)
        bucket = s3.Bucket(bucket_name)

        # Get all leaf folders and randomly select num_files
        leaf_folders = set()
        for obj in bucket.objects.filter(Prefix=prefix):
            leaf_folder = os.path.dirname(obj.key)
            if leaf_folder != prefix:
                leaf_folders.add(leaf_folder)
        leaf_folders = set(random.sample(list(leaf_folders), min(num_files, len(leaf_folders))))

        self.logger.info(f"=========================")
        self.logger.info(f"Found {len(leaf_folders)} leaf folders")
        self.logger.info(f"=========================\n")
        
        # Sort leaf folders and create numbered local folders
        selected_folders = sorted(leaf_folders)

        # Initialize progress bars
        pbar_folders = tqdm(enumerate(selected_folders, 1), desc='Processing folders', unit='folder')

        for folder_num, folder in pbar_folders:
            # Create numbered local folder
            numbered_folder = os.path.join(local_dir, str(folder_num))
            os.makedirs(numbered_folder, exist_ok=True)
            
            # Get all objects in the selected folder
            objects = list(bucket.objects.filter(Prefix=folder))
            
            # Download each object
            for obj in objects:
                local_file_path = os.path.join(numbered_folder, os.path.basename(obj.key))
                bucket.download_file(obj.key, local_file_path)

        pbar_folders.close()

    def upload_to_s3(self, local_dir = None, s3_uri = None):
        """Upload model-data folders from local directory to S3"""
        
        assert local_dir is not None, "local_dir is required"
        assert s3_uri is not None, "s3_uri is required"
        
        # Parse S3 URI to get bucket and prefix
        s3_parts = s3_uri.replace("s3://", "").split("/")
        bucket = s3_parts[0]
        prefix = "/".join(s3_parts[1:]) if len(s3_parts) > 1 else ""
        
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Find all model-data folders
        model_data_folders = [d for d in os.listdir(local_dir) 
                             if os.path.isdir(os.path.join(local_dir, d)) and 
                             d.startswith('model-data-')]
        
        if not model_data_folders:
            self.logger.warning(f"No model-data folders found in {local_dir}")
            return
        
        self.logger.info(f"=========================")
        self.logger.info(f"Found model-data folders: {model_data_folders}")
        self.logger.info(f"Uploading to {s3_uri}")
        self.logger.info(f"=========================\n")
        
        # Count total files for progress bar
        total_files = sum(len(files) for folder in model_data_folders 
                         for _, _, files in os.walk(os.path.join(local_dir, folder)))
        
        with tqdm(total=total_files, desc="Uploading files", unit='file') as pbar:
            # Walk through each model-data folder
            for folder in model_data_folders:
                folder_path = os.path.join(local_dir, folder)

                # self.logger.info(f"=========================    ")
                # self.logger.info(f"folder_path: {folder_path}")
                # self.logger.info(f"=========================\n")
                
                
                for root, _, files in os.walk(folder_path):
                    for filename in files:
                        # Get full local file path
                        file_path = os.path.join(root, filename)
                        
                        # Calculate relative path from the model-data folder
                        relative_path = os.path.relpath(file_path, folder_path)
                        
                        # Create base S3 path
                        base_s3_path = f"{prefix}/{os.path.basename(local_dir)}"

                        # Construct S3 key with prefix and folder name
                        s3_key = f"{base_s3_path}/{folder}/{relative_path}".replace("\\", "/")
                        
                        # self.logger.info(f"=========================    ")
                        # self.logger.info(f"prefix: {prefix}")
                        # self.logger.info(f"base_s3_path: {base_s3_path}")
                        # self.logger.info(f"s3_key: {s3_key}")   
                        # self.logger.info(f"file_path: {file_path}")
                        # self.logger.info(f"bucket: {bucket}")
                        # self.logger.info(f"=========================\n")
                        # continue
                        
                        # Upload file
                        try:
                            s3_client.upload_file(file_path, bucket, s3_key)
                            pbar.update(1)
                        except Exception as e:
                            self.logger.error(f"Error uploading {file_path}: {str(e)}")

      
    
if __name__ == "__main__":
    
    data_generator = DataGenerator()
    
    # s3_uri = "s3://occupancy-dataset/occ-dataset/vineyards/gallo/"

    # fetch data from s3
    # data_generator.fetch_data_from_s3(s3_uri, "data/aws-data/gallo", num_files=300)

    # process s3 data and move to raw-data folder
    # data_generator.s3_data_to_raw_data("data/gallo/aws-data", "data/gallo/raw-data")

    # # process raw data and move to model-data folder
    # data_generator.raw_data_to_model_data("data/gallo/raw-data", "data/gallo/model-data-1920x1080")
    
    # rescale images to 640x480
    # data_generator.rescale_images(src_folder="data/gallo/model-data-1920x1080", dst_folder="data/gallo/model-data-640x480", h=480, w=640)

    # upload to s3
    data_generator.upload_to_s3(local_dir="data/RJM", 
                                s3_uri="s3://occupancy-dataset/bev-dataset")
