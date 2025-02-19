import boto3
import logging
from typing import List, Dict, Any, Tuple
import yaml
from pathlib import Path
import argparse
import sys
import random
import pyzed.sl as sl
import cv2
import numpy as np
import tempfile
import os
from tqdm import tqdm
import json  # added for JSON processing of the index file

from scripts.helpers import get_logger


class EvalDataS3:

    @staticmethod
    def get_unique_svo_folders(s3_uri: str) -> List[str]:
        """
        Recursively finds all folders in the given S3 URI that contain at least one .svo file.

        Args:
            s3_uri (str): AWS S3 URI, e.g. "s3://bucket-name/path/to/folder"

        Returns:
            List[str]: A sorted list of unique S3 folder URIs that contain .svo files

        Raises:
            ValueError: If the provided s3_uri is not a valid S3 URI
            Exception: If an error occurs when listing S3 objects
        """
        # validate s3_uri starts with 's3://'
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"provided uri '{s3_uri}' is not a valid s3 uri")

        try:
            # parse bucket and prefix from the S3 URI
            uri_without_scheme = s3_uri[5:]
            parts = uri_without_scheme.split('/', 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            # ensure prefix ends with '/' if non-empty
            if prefix and not prefix.endswith('/'):
                prefix += '/'
        except Exception as parse_error:
            logger.error(f"error parsing s3 uri: {s3_uri}. details: {parse_error}", exc_info=True)
            raise

        folders_with_svo = set()
        try:
            # create s3 client
            s3_client = boto3.client('s3')
            # use paginator to handle large number of objects
            paginator = s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj.get("Key", "")
                    if key.lower().endswith(".svo"):
                        # get the folder path by removing the filename
                        folder_path = f"s3://{bucket}/{key.rsplit('/', 1)[0]}"
                        folders_with_svo.add(folder_path)
        except Exception as list_error:
            logger.error(
                f"error listing s3 objects for bucket '{bucket}' with prefix '{prefix}'. details: {list_error}",
                exc_info=True
            )
            raise Exception("failed to list folders containing .svo files from s3") from list_error

        # return sorted list for consistent output
        return sorted(list(folders_with_svo))

    def is_valid_folder(folder: str, farms_to_sample: List[str]) -> Tuple[bool, str]:
        """
        Checks if a folder contains a valid farm name from the list of farms to sample.

        Args:
            folder (str): The folder path to check

        Returns:
            Tuple[bool, str]: True and the farm name if the folder contains a valid farm name, False and an empty string otherwise.
        """
        for farm in farms_to_sample:
            if farm in folder:
                return True, farm
        return False, ""

    


    @staticmethod
    def sample_svo_from_folders(s3_uri: str, 
                                farms_to_sample: List[str], 
                                local_dir: str, 
                                num_frames: int = 20) -> Dict[str, str]:
        """
        Downloads one SVO file from each specified farm folder while maintaining the directory structure.

        Args:
            s3_uri (str): AWS S3 URI, e.g. "s3://bucket-name/path/to/folder"
            farms_to_sample (List[str]): List of farm names to sample from
            local_dir (str): Local directory path where files should be downloaded

        Returns:
            Dict[str, str]: Mapping of farm names to downloaded file paths

        Raises:
            ValueError: If the provided parameters are invalid
            Exception: If an error occurs during S3 operations
        """
        # get all folders containing svo files
        all_folders = EvalDataS3.get_unique_svo_folders(s3_uri)

        valid_folders = []
        for folder in all_folders:
            is_valid, farm_name = EvalDataS3.is_valid_folder(folder, farms_to_sample)
            if is_valid:
                valid_folders.append((folder, farm_name))

        # logger.info(f"───────────────────────────────")
        # logger.info(f"found {len(valid_folders)} valid folders")
        # logger.info(f"valid folders: {valid_folders}")
        # logger.info(f"───────────────────────────────")

        downloaded_files = {}
        s3_client = boto3.client('s3')

        from tqdm import tqdm
        downloaded_files = {}
        s3_client = boto3.client('s3')

        for folder, farm_name in valid_folders:
            folder_name = folder.replace(f"s3://sg-new-data/dairy_farm/{farm_name}/", "")
            dest_folder = Path(Path(local_dir) / farm_name / folder_name)
            dest_folder.mkdir(parents=True, exist_ok=True)

            svo_uris, _ = EvalDataS3.get_svo_uris_in_folder(folder)
            if not svo_uris:
                logger.warning(f"No SVO files found in folder: {folder}")
                continue

            for svo_uri in svo_uris:
                dest_URI_base = "s3://occupancy-dataset/svo-images"
                dest_URI_suffix = f"{farm_name}/{folder_name}/{Path(svo_uri).name}"
                dest_URI = f"{dest_URI_base.rstrip('/')}/{dest_URI_suffix.lstrip('/')}"

                # logger.warning(f"───────────────────────────────")
                # logger.warning(f"dest_URI_base: {dest_URI_base}")
                # logger.warning(f"dest_URI_suffix: {dest_URI_suffix}")
                # logger.warning(f"dest_URI: {dest_URI}")
                # logger.warning(f"───────────────────────────────")

                EvalDataS3.process_svo_uri(svo_uri, dest_URI, num_frames)

    @staticmethod
    def get_svo_uris_in_folder(s3_uri: str) -> Tuple[List[str], int]:
        """
        Recursively finds all .svo files in the given S3 URI folder.

        Args:
            s3_uri (str): AWS S3 URI, e.g. "s3://bucket-name/path/to/folder"

        Returns:
            Tuple[List[str], int]: A tuple containing:
                - List of S3 keys for all .svo files
                - Total number of .svo files found

        Raises:
            ValueError: If the provided s3_uri is not a valid S3 URI
            Exception: If an error occurs when listing S3 objects
        """
        # validate s3_uri starts with 's3://'
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"provided uri '{s3_uri}' is not a valid s3 uri")

        try:
            # parse bucket and prefix from the S3 URI
            uri_without_scheme = s3_uri[5:]
            parts = uri_without_scheme.split('/', 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            # ensure prefix ends with '/' if non-empty
            if prefix and not prefix.endswith('/'):
                prefix += '/'

        except Exception as parse_error:
            logger.error(f"error parsing s3 uri: {s3_uri}. details: {parse_error}", exc_info=True)
            raise

        svo_uris = []
        try:
            # create s3 client
            s3_client = boto3.client('s3')
            # use paginator to handle large number of objects
            paginator = s3_client.get_paginator("list_objects_v2")
            
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj.get("Key", "")
                    if key.lower().endswith(".svo"):
                        svo_uris.append(f"s3://{bucket}/{key}")

        except Exception as list_error:
            logger.error(
                f"error listing s3 objects for bucket '{bucket}' with prefix '{prefix}'. details: {list_error}",
                exc_info=True
            )
            raise Exception("failed to list .svo files from s3") from list_error

        total_files = len(svo_uris)
        logger.info(f"found {total_files} .svo files in {s3_uri}")
        
        return svo_uris, total_files

    @staticmethod
    def process_svo_uri(svo_uri: str, output_s3_uri: str, num_frames: int = 20) -> None:
        """
        Downloads an SVO file, extracts frames, processes them, and uploads to S3.
        
        After successful processing, the SVO file's URI is recorded in the index file at
        'index-s3/eval-data-s3.json' so that it is not processed again.

        Args:
            svo_uri (str): S3 URI of the source SVO file
            output_s3_uri (str): S3 URI where processed images should be uploaded
            num_frames (int): Number of frames to sample from the SVO file

        Raises:
            ValueError: If the provided URIs are invalid
            Exception: If an error occurs during processing
        """

        logger.info(f"───────────────────────────────")
        logger.info(f"Processing {svo_uri}...")
        logger.info(f"───────────────────────────────")

        # check processed index json file to skip already processed SVO files
        index_dir = Path("index-s3")
        index_file = index_dir / "eval-data-s3.json"
        try:
            index_dir.mkdir(parents=True, exist_ok=True)
            if index_file.exists():
                with index_file.open("r") as f:
                    processed_files = json.load(f)
                if not isinstance(processed_files, list):
                    logger.warning(f"index file {index_file} does not contain a list; reinitializing it.")
                    processed_files = []
            else:
                processed_files = []
        except Exception as io_err:
            logger.error(f"error handling index file '{index_file}': {io_err}", exc_info=True)
            processed_files = []  # default to empty list if there is an error

        if svo_uri in processed_files:
            logger.warning(f"svo file {svo_uri} already processed, skipping further processing.")
            return

        # create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            svo_filename = Path(svo_uri).name
            local_svo_path = temp_dir_path / svo_filename

            try:
                # download svo file
                bucket = svo_uri.split('/')[2]
                key = '/'.join(svo_uri.split('/')[3:])
                s3_client = boto3.client('s3')
                # logger.info(f"downloading {svo_uri} to {local_svo_path}")
                s3_client.download_file(bucket, key, str(local_svo_path))

                # initialize zed camera
                init_params = sl.InitParameters()
                init_params.set_from_svo_file(str(local_svo_path))
                zed = sl.Camera()
                status = zed.open(init_params)
                if status != sl.ERROR_CODE.SUCCESS:
                    raise Exception(f"failed to open svo file: {status}")

                # ensure that the camera closes even if processing fails
                try:
                    total_frames = zed.get_svo_number_of_frames()
                    if total_frames < num_frames:
                        logger.warning(f"svo file has fewer frames ({total_frames}) than requested ({num_frames})")
                        frame_indices = range(total_frames)
                    else:
                        frame_indices = sorted(random.sample(range(total_frames), num_frames))

                    image = sl.Mat()

                    for idx, frame_num in enumerate(frame_indices):
                        zed.set_svo_position(frame_num)
                        if zed.grab() == sl.ERROR_CODE.SUCCESS:
                            image_left = sl.Mat()
                            zed.retrieve_image(image_left, sl.VIEW.LEFT)
                            img_opencv_left = image_left.get_data()
                            img_resized_left = cv2.resize(img_opencv_left, (640, 480))
                            temp_img_path_left = temp_dir_path / f"frame_{frame_indices[idx]}_left.jpg"
                            cv2.imwrite(str(temp_img_path_left), img_resized_left)
                            output_key_left = f"{output_s3_uri.split('s3://')[-1]}/frame_{frame_indices[idx]}_left.jpg"
                            output_bucket = output_s3_uri.split('/')[2]
                            s3_client.upload_file(
                                str(temp_img_path_left),
                                output_bucket,
                                '/'.join(output_key_left.split('/')[1:])
                            )
                            # logger.info(f"uploaded left frame {idx} to {output_key_left}")

                            image_right = sl.Mat()
                            zed.retrieve_image(image_right, sl.VIEW.RIGHT)
                            img_opencv_right = image_right.get_data()
                            img_resized_right = cv2.resize(img_opencv_right, (640, 480))
                            temp_img_path_right = temp_dir_path / f"frame_{frame_indices[idx]}_right.jpg"
                            cv2.imwrite(str(temp_img_path_right), img_resized_right)
                            output_key_right = f"{output_s3_uri.split('s3://')[-1]}/frame_{frame_indices[idx]}_right.jpg"
                            s3_client.upload_file(
                                str(temp_img_path_right),
                                output_bucket,
                                '/'.join(output_key_right.split('/')[1:])
                            )
                            # logger.info(f"uploaded right frame {idx} to {output_key_right}")
                        else:
                            logger.warning(f"failed to grab frame {frame_num} from {svo_uri}")
                finally:
                    zed.close()

            except Exception as e:
                logger.error(f"error processing svo file {svo_uri}: {str(e)}", exc_info=True)
                raise

            logger.info(f"completed processing {svo_uri}")

        # update index file after successful processing
        try:
            processed_files.append(svo_uri)
            with index_file.open("w") as f:
                json.dump(processed_files, f, indent=2)
            logger.info(f"added svo file {svo_uri} to index file '{index_file}'.")
        except Exception as write_err:
            logger.error(f"error updating index file '{index_file}': {write_err}", exc_info=True)


if __name__ == "__main__":
    
    logger = get_logger("eval-data-s3")
    parser = argparse.ArgumentParser(
        description="Recursively list .svo files from a given AWS S3 URI"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/eval-data-s3.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()

    # load configuration(s) from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        dairy_folder_URI = config.get('dairy_folder_URI')
        farms_to_sample = config.get('farms_to_sample')
        svo_frames_to_sample = config.get('svo_frames_to_sample', 20)

    assert dairy_folder_URI is not None, "dairy_folder_URI is not set in the config file"
    assert farms_to_sample is not None, "farms_to_sample is not set in the config file"
    assert svo_frames_to_sample is not None, "svo_frames_to_sample is not set in the config file"
    
    # process yaml config
    s3_uri = dairy_folder_URI


    # # get folders with svo files
    # folders_with_svo = EvalDataS3.get_unique_svo_folders(s3_uri)
   
    # logger.info(f"───────────────────────────────")
    # logger.info(f"found {len(folders_with_svo)} folders with svo files.")
    # for idx, folder in enumerate(folders_with_svo):
    #     logger.info(f"{idx}: {folder}")
    # logger.info(f"───────────────────────────────")
    

    # sample and download svo files
    EvalDataS3.sample_svo_from_folders(
        s3_uri=s3_uri,
        farms_to_sample=farms_to_sample,
        local_dir="eval-data/svo-files",
        num_frames=svo_frames_to_sample
    )
