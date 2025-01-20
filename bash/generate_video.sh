#! /bin/bash

IMAGE_DIR=debug/9/left-imgs
OUTPUT_VIDEO=debug/9/videos/left.mp4
FPS=10

if [ -z "$IMAGE_DIR" ] || [ -z "$OUTPUT_VIDEO" ]; then
  echo "Usage: $0 <image_directory> <output_video_path> [fps]"
  exit 1
fi

if [ -z "$FPS" ]; then
    FPS=10
fi

python3 -m scripts.get_video "$IMAGE_DIR" "$OUTPUT_VIDEO" --fps "$FPS" 