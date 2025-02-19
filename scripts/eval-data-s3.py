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
    def get_svo_files_URIs(s3_uri: str) -> Tuple[int, List[str]]:
        """
        Recursively iterates over all files in the given AWS S3 URI and returns 
        the S3 URIs for all the files with a '.svo' extension.

        Args:
            s3_uri (str): AWS S3 URI, e.g. "s3://bucket-name/path/to/folder"

        Returns:
            List[str]: A list of S3 URIs corresponding to .svo files.

        Raises:
            ValueError: If the provided s3_uri is not a valid S3 URI.
            Exception: If an error occurs when listing S3 objects.
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

        svo_files = []
        try:
            # create s3 client
            s3_client = boto3.client('s3')
            # use paginator to recursively list objects under the prefix
            paginator = s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj.get("Key", "")
                    # append only .svo files
                    if key.lower().endswith(".svo"):
                        svo_files.append(f"s3://{bucket}/{key}")
        except Exception as list_error:
            logger.error(
                f"error listing s3 objects for bucket '{bucket}' with prefix '{prefix}'. details: {list_error}",
                exc_info=True
            )
            raise Exception("failed to list .svo files from s3") from list_error

        return len(svo_files), svo_files

    @staticmethod
    def get_unique_parent_directories(svo_uris: List[str]) -> List[str]:
        """
        Extracts unique parent directories from a list of SVO file URIs.

        Args:
            svo_uris (List[str]): List of S3 URIs for SVO files
                                 e.g. ["s3://bucket/path/to/file1.svo", 
                                      "s3://bucket/path/to/file2.svo"]

        Returns:
            List[str]: List of unique parent directory URIs
                      e.g. ["s3://bucket/path/to"]

        Raises:
            ValueError: If any URI in the list is not a valid S3 URI
        """
        # validate and extract parent directories
        parent_dirs = set()
        for uri in svo_uris:
            if not uri.startswith("s3://"):
                raise ValueError(f"invalid s3 uri format: {uri}")
            
            # remove the filename to get parent directory
            parent_dir = uri.rsplit('/', 1)[0]
            parent_dirs.add(parent_dir)
        
        # convert set to sorted list for consistent output
        return sorted(list(parent_dirs))


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


    num_svo_files, svo_files = EvalDataS3.get_svo_files_URIs(s3_uri)
    
    logger.info(f"───────────────────────────────")
    logger.info(f"found {num_svo_files} .svo files.")
    logger.info(f"───────────────────────────────")
    
    unique_parent_dirs = EvalDataS3.get_unique_parent_directories(svo_files)
    
    logger.info(f"───────────────────────────────")
    logger.info(f"found {len(unique_parent_dirs)} unique parent directories.")
    logger.info(f"───────────────────────────────")

    for parent_dir in unique_parent_dirs:
        logger.info(f"processing parent directory: {parent_dir}")
        
