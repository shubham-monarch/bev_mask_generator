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

    # Pass dummy values to bypass assertions
    data_generator_s3 = DataGeneratorS3(
        src_URIs=URIs, 
        dest_folder=None,
        index_json="dummy",     # Dummy value to bypass assert
        color_map="dummy",      # Dummy value to bypass assert
        crop_bb={"x": 0, "y": 0, "z": 0, "w": 1, "h": 1, "d": 1},  # Dummy value to bypass assert
        nx=1,                   # Dummy value to bypass assert
        nz=1                    # Dummy value to bypass assert
    )
    
    leaf_folder_cnt = len(data_generator_s3.get_leaf_folders())
    
    logger.info(f"=======================")
    logger.info(f"URI: {args.uri}")
    logger.info(f"Total leaf folders: {leaf_folder_cnt}")
    logger.info(f"=======================\n")
    