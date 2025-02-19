import boto3
import logging
from typing import List, Dict, Any, Tuple
import yaml
from pathlib import Path
import argparse
import sys



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

    # load configuration from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        dairy_folder_URI = config.get('dairy_folder_URI')

    assert dairy_folder_URI is not None, "dairy_folder_URI is not set in the config file"
    s3_uri = dairy_folder_URI


    # get folders with svo files
    folders_with_svo = EvalDataS3.get_unique_svo_folders(s3_uri)
    logger.info(f"───────────────────────────────")
    logger.info(f"found {len(folders_with_svo)} folders with svo files.")
    logger.info(f"───────────────────────────────")

    for folder in folders_with_svo:
        logger.info(f"processing folder: {folder}")
    
    
        
