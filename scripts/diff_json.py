#! /usr/bin/env python3

import argparse
import json
import logging
import sys
from typing import Any, Dict

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("diff_json")


def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file from the given file path.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Loaded JSON data.

    Raises:
        Exception: if the file cannot be read or parsed.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as exc:
        logger.error(f"failed to read or parse JSON file '{file_path}': {exc}")
        raise

    if not isinstance(data, dict):
        error_msg = f"file '{file_path}' does not contain a JSON object."
        logger.error(error_msg)
        raise ValueError(error_msg)

    return data


def diff_keys(source_data: Dict[str, Any], target_data: Dict[str, Any], key_filter: str = None) -> None:
    """
    Compare keys of target JSON against source JSON and log the differences.

    Args:
        source_data (dict): The JSON data from the source file.
        target_data (dict): The JSON data from the target file.
    """
    source_keys = set(source_data.keys())
    target_keys = set(target_data.keys())
    missing_keys = target_keys - source_keys

    # Filter out keys containing "open_sky"
    if key_filter:
        missing_keys = [key for key in missing_keys if key_filter in key]

    
    if missing_keys:
        logger.info("keys in target JSON that are not in source JSON:")
        for key in sorted(missing_keys):
            logger.info(f"- {key}")
    else:
        logger.info("no keys found in target JSON that are not in source JSON.")


def main() -> None:
    """
    Main function to parse arguments and compare JSON files.

    Expects two positional arguments:
        source_json: Path to the source JSON file.
        target_json: Path to the target JSON file.
    """
    parser = argparse.ArgumentParser(
        description="Compare two JSON files and list out the keys in the target JSON file that are not in the source JSON file."
    )
    parser.add_argument("--s", type=str, help="Path to the source JSON file")
    parser.add_argument("--t", type=str, help="Path to the target JSON file")
    args = parser.parse_args()

    try:
        source_data = load_json_file(args.s)
    except Exception:
        sys.exit(1)

    try:
        target_data = load_json_file(args.t)
    except Exception:
        sys.exit(1)

    diff_keys(source_data, target_data, "open_sky")


if __name__ == "__main__":
    main() 