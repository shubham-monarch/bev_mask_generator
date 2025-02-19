import boto3
import logging
from typing import List, Dict, Any, Tuple
import yaml
from pathlib import Path
import argparse
import sys
import random


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
    def sample_svo_from_folders(s3_uri: str, farms_to_sample: List[str], local_dir: str) -> Dict[str, str]:
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

        logger.info(f"───────────────────────────────")
        logger.info(f"found {len(valid_folders)} valid folders")
        logger.info(f"valid folders: {valid_folders}")
        logger.info(f"───────────────────────────────")

        downloaded_files = {}
        s3_client = boto3.client('s3')

        from tqdm import tqdm

        downloaded_files = {}
        s3_client = boto3.client('s3')

        with tqdm(total=len(valid_folders), desc="Downloading SVO files") as pbar:
            for folder, farm_name in valid_folders:
                
                folder_name = folder.replace(f"s3://sg-new-data/dairy_farm/{farm_name}/", "")
                dest_folder = Path(Path(local_dir) / farm_name / folder_name)
                dest_folder.mkdir(parents=True, exist_ok=True)

                svo_uris, _ = EvalDataS3.get_svo_uris_in_folder(folder)
                if not svo_uris:
                    logger.warning(f"No SVO files found in folder: {folder}")
                    continue

                svo_uri = random.choice(svo_uris)

                logger.warning(f"───────────────────────────────") 
                logger.warning(f"dest_folder: {dest_folder.as_posix()}")
                logger.warning(f"downloading {svo_uri}")
                logger.warning(f"───────────────────────────────")
                
                dest_file = Path(dest_folder / Path(svo_uri).name)

                try:
                    bucket = svo_uri.split('/')[2]
                    key = '/'.join(svo_uri.split('/')[3:])
                    s3_client.download_file(
                        Bucket=bucket,
                        Key=key,
                        Filename=str(dest_file)
                    )
                    downloaded_files[farm_name] = str(dest_file)
                    logger.info(f"Downloaded {svo_uri} to {dest_file}")
                except Exception as e:
                    logger.error(f"Error downloading {svo_uri}: {e}", exc_info=True)

              
                pbar.update(1)

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

    assert dairy_folder_URI is not None, "dairy_folder_URI is not set in the config file"
    assert farms_to_sample is not None, "farms_to_sample is not set in the config file"
    
    # process yaml config
    s3_uri = dairy_folder_URI


    # get folders with svo files
    folders_with_svo = EvalDataS3.get_unique_svo_folders(s3_uri)
   
    logger.info(f"───────────────────────────────")
    logger.info(f"found {len(folders_with_svo)} folders with svo files.")
    for idx, folder in enumerate(folders_with_svo):
        logger.info(f"{idx}: {folder}")
    logger.info(f"───────────────────────────────")
    

    # sample and download svo files
    EvalDataS3.sample_svo_from_folders(
        s3_uri=s3_uri,
        farms_to_sample=farms_to_sample,
        local_dir="eval-data/svo-files"
    )
