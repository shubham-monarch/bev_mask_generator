"""Demo video data generation script"""

#! /usr/bin/env python3

from typing import List, Tuple
from pathlib import Path
from tqdm import tqdm
import random

from scripts.data_generator_s3 import DataGeneratorS3, LeafFolder
from scripts.logger import get_logger
import boto3

def process_leaf_parent_folder(leaf_uri: str) -> Tuple[List[str], List[str]]:
    """
    Finds all 'left.jpg' files and their corresponding 'seg-mask-rgb.png' files in a given S3 URI path.

    This function searches recursively within the provided S3 URI for all 'left.jpg' files.
    It then constructs the corresponding 'seg-mask-rgb.png' URI by replacing 'occ-dataset' with 'bev-dataset'
    and 'left.jpg' with 'seg-mask-rgb.png'.

    Args:
        leaf_uri: The S3 URI path to search within.

    Returns:
        A tuple containing two lists:
            - A list of S3 URIs for all 'left.jpg' files found.
            - A list of S3 URIs for the corresponding 'seg-mask-rgb.png' files.
    """
    logger = get_logger("download_left_images")
    
    # handle both string and Path objects, ensure we're working with a string
    leaf_uri = str(leaf_uri)
    
    # get the bucket and prefix from the s3 uri
    if not leaf_uri.startswith('s3://'):
        raise ValueError(f"Invalid S3 URI format: {leaf_uri}. Must start with 's3://'")
        
    s3_path = leaf_uri.replace("s3://", "")
    parts = s3_path.split("/", 1)  # split only on first '/' to separate bucket and prefix
    
    if len(parts) < 2:
        raise ValueError(f"Invalid S3 URI format: {leaf_uri}. Must include bucket and prefix")
        
    bucket = parts[0]
    prefix = parts[1]
    
    # logger.info("───────────────────────────────")
    # logger.info(f"searching in bucket: {bucket}")
    # logger.info(f"with prefix: {prefix}")
    # logger.info("───────────────────────────────")

    # initialize s3 client
    s3_client = boto3.client('s3')
    
    left_img_URIs = []
    bev_img_URIs = []

    paginator = s3_client.get_paginator('list_objects_v2')
    
    # paginate through results since s3 limits to 1000 objects per request
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            if obj['Key'].endswith('left.jpg'):
                left_image_uri = f"s3://{bucket}/{obj['Key']}"
                left_img_URIs.append(left_image_uri)
                
                bev_image_uri = left_image_uri.replace("occ-dataset", "bev-dataset")
                bev_image_uri = bev_image_uri.replace("left.jpg", "seg-mask-rgb.png")
                bev_img_URIs.append(bev_image_uri)

    # logger.info("───────────────────────────────")
    # logger.info(f"total left images found: {len(left_img_URIs)}")
    # logger.info(f"total bev images found: {len(bev_img_URIs)}")
    # logger.info("───────────────────────────────")
    
    return left_img_URIs, bev_img_URIs

def get_formatted_name_from_uri(uri: str) -> str:
    """
    Extracts and formats a name from an S3 URI path.
    
    Args:
        uri: S3 URI of the form 
            's3://occupancy-dataset/bev-dataset/vineyards/gallo/2024_06_07_utc/svo_files/front_2024-06-04-10-24-57.svo/1398_to_1540/'

    Returns:
        Formatted string like: 'gallo_front_2024-06-04-10-24-57_1398_to_1540'
    """
    # remove s3:// prefix and split path
    path_parts = uri.replace('s3://', '').split('/')
    
    # extract relevant parts
    vineyard = path_parts[3]  # 'gallo'
    camera_timestamp = path_parts[6].replace('.svo', '')  # 'front_2024-06-04-10-24-57'
    frame_range = path_parts[7]  # '1398_to_1540'
    
    # combine parts with underscores
    formatted_name = f"{vineyard}_{camera_timestamp}_{frame_range}"
    
    return formatted_name

def download_s3_file(s3_uri: str, local_path: Path, file_type: str = "jpg") -> None:
    """
    Downloads a file from S3 to a local destination with sequential numbering.
    
    Args:
        s3_uri: The S3 URI of the file to download
        local_path: The local path where the file should be saved
        file_type: The file extension to use (default: "jpg")
    """
    logger = get_logger("download_s3_file")
    try:
        bucket_name, key = s3_uri.replace("s3://", "").split("/", 1)
        s3 = boto3.client('s3')
        # logger.info(f"downloading {s3_uri} to {local_path}...")
        s3.download_file(bucket_name, key, str(local_path))
        # logger.info(f"successfully downloaded {s3_uri} to {local_path}")
    except Exception as e:
        logger.error(f"───────────────────────────────")
        logger.error(f"Failed to download {s3_uri} to {local_path}: {e}")
        logger.error(f"───────────────────────────────")

def download_image_sequence(image_uris: List[str], output_dir: Path, file_type: str) -> None:
    """
    Downloads a sequence of images from S3 URIs and saves them with sequential numbering.
    
    Args:
        image_uris: List of S3 URIs for the images
        output_dir: Directory where images should be saved
        file_type: File extension for the saved images ("jpg" or "png")
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, uri in enumerate(image_uris):
        local_path = output_dir / f"{idx:04d}.{file_type}"
        download_s3_file(uri, local_path)

def process_leaf_folders(src_URI: List[str]) -> None:
    logger = get_logger("process_leaf_folder")
    
    leaf_folder_URIs = DataGeneratorS3.get_leaf_folders(src_URI)
    random.shuffle(leaf_folder_URIs)

    

    logger.warning("───────────────────────────────")
    logger.warning(f"len(leaf_folder_URIs): {len(leaf_folder_URIs)}")
    logger.warning("───────────────────────────────")

    processed_parent_folders = {}

    for idx, leaf_folder_uri in enumerate(tqdm(leaf_folder_URIs, desc="Processing leaf folders")):
        # if idx > 0:
        #     break
        
        # handle s3 uri parent path properly
        parent_folder_uri = str(Path(leaf_folder_uri.replace('s3://', '')).parent)
        parent_folder_uri = f"s3://{parent_folder_uri}"
        
        logger.info("───────────────────────────────")
        logger.info(f"leaf_folder_uri: {leaf_folder_uri}")
        logger.info(f"parent_folder_uri: {parent_folder_uri}")
        logger.info("───────────────────────────────")

        if parent_folder_uri not in processed_parent_folders:
            left_img_URIs, bev_img_URIs = process_leaf_parent_folder(str(parent_folder_uri))
            processed_parent_folders[parent_folder_uri] = True

            # logger.info(f"───────────────────────────────")
            # logger.info(f"left_img_URIs: {left_img_URIs}")
            # logger.info(f"bev_img_URIs: {bev_img_URIs}")
            # logger.info(f"───────────────────────────────\n")

    
            logger.info(f"left_img_URI: {left_img_URIs[0]}")
            logger.info(f"bev_img_URI: {bev_img_URIs[0]}")

            formatted_name = get_formatted_name_from_uri(str(parent_folder_uri))
            left_img_folder = Path(f"demo/{formatted_name}/left_images")
            bev_img_folder = Path(f"demo/{formatted_name}/bev_images")
            
            left_img_folder.mkdir(parents=True, exist_ok=True)
            bev_img_folder.mkdir(parents=True, exist_ok=True)

            # download the images
            for idx, left_img_uri in enumerate(left_img_URIs):
                local_path = left_img_folder / f"{idx:04d}.jpg"
                download_s3_file(left_img_uri, local_path)

            for idx, bev_img_uri in enumerate(bev_img_URIs):
                local_path = bev_img_folder / f"{idx:04d}.png"
                download_s3_file(bev_img_uri, local_path)

        # else:
        #     logger.info("───────────────────────────────")
        #     logger.info(f"parent folder already processed: {parent_folder_uri}")
        #     logger.info("───────────────────────────────")

if __name__ == "__main__":
    logger = get_logger("demo")

    src_URI:List[str] = ["s3://occupancy-dataset/occ-dataset/vineyards/RJM/"]

    process_leaf_folders(src_URI)
