#! /usr/bin/env python3

import cv2
import os
from tqdm import tqdm
import argparse
import re
from scripts.logger import get_logger

logger = get_logger("get_video")

def create_video_from_images(image_dir: str, output_video_path: str, fps: int = 10) -> None:
    """
    Creates a video from a sequence of images in a directory.

    Args:
        image_dir (str): The path to the directory containing the images.
        output_video_path (str): The path where the output video file will be saved.
        fps (int, optional): The frames per second of the output video. Defaults to 10.
    
    Raises:
        FileNotFoundError: If the image directory does not exist.
        ValueError: If no images are found in the directory.
        Exception: If there is an error during video creation.
    """
    if not os.path.exists(image_dir):
        logger.error(f"Image directory not found: {image_dir}")
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        logger.error(f"No images found in directory: {image_dir}")
        raise ValueError(f"No images found in directory: {image_dir}")
    
    # use natural sorting to handle numeric portions of filenames correctly
    def natural_sort_key(s: str) -> list:
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)]
    
    image_files.sort(key=natural_sort_key)  # sort using natural sort for correct numeric ordering
    
    logger.info("───────────────────────────────")
    logger.info(f"image_files: {image_files}")    
    logger.info("───────────────────────────────")


    try:
        # create the output directory if it doesn't exist
        output_dir = os.path.dirname(output_video_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

        # Read the first image to get dimensions
        first_image_path = os.path.join(image_dir, image_files[0])
        first_image = cv2.imread(first_image_path)
        height, width, _ = first_image.shape
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 output
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        for image_file in tqdm(image_files, desc="Creating video"):
            image_path = os.path.join(image_dir, image_file)
            frame = cv2.imread(image_path)
            video_writer.write(frame)
        
        video_writer.release()
        logger.info(f"Video created successfully at: {output_video_path}")
    
    except Exception as e:
        logger.error(f"Error creating video: {e}", exc_info=True)
        raise Exception(f"Error creating video: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a video from a directory of images.")
    parser.add_argument("image_dir", type=str, help="Path to the directory containing images.")
    parser.add_argument("output_video_path", type=str, help="Path to save the output video.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the output video.")

    args = parser.parse_args()

    try:
        create_video_from_images(args.image_dir, args.output_video_path, args.fps)
    except Exception as e:
        logger.error(f"Failed to create video: {e}")