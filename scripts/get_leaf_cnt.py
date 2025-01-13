#! /usr/bin/env python3

from scripts.data_generator_s3 import DataGeneratorS3
from scripts.logger import get_logger
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count leaf folders in S3 bucket')
    parser.add_argument('--uri', type=str, default="s3://occupancy-dataset/occ-dataset/dairy/",
                      help='S3 URI to scan')
    
    args = parser.parse_args()
    
    # =================
    # counting processed frames
    # =================
    
    logger = get_logger("get_leaf_cnt")

    URIs = [args.uri]
    
    leaf_folder_cnt = len(DataGeneratorS3.get_leaf_folders(URIs))
    
    logger.info(f"=======================")
    logger.info(f"URI: {args.uri}")
    logger.info(f"Total leaf folders: {leaf_folder_cnt}")
    logger.info(f"=======================\n")
    