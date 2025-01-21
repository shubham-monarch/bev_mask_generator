#! /bin/bash

# left-imgs
IMAGE_DIR=debug/10/left-imgs
OUTPUT_VIDEO=debug/10/videos/left.mp4

# bev-poles
# IMAGE_DIR=debug/10/bev-poles
# OUTPUT_VIDEO=debug/10/videos/bev-poles.mp4

# bev-all
# IMAGE_DIR=debug/9/bev-all
# OUTPUT_VIDEO=debug/9/videos/bev-all.mp4

# seg-pcd-projection
# IMAGE_DIR=debug/10/seg-pcd-projection
# OUTPUT_VIDEO=debug/10/videos/seg-pcd-projection.mp4

# rgb-pcd-projection
# IMAGE_DIR=debug/10/rgb-pcd-projection
# OUTPUT_VIDEO=debug/10/videos/rgb-pcd-projection.mp4


FPS=10

if [ -z "$IMAGE_DIR" ] || [ -z "$OUTPUT_VIDEO" ]; then
  echo "Usage: $0 <image_directory> <output_video_path> [fps]"
  exit 1
fi

if [ -z "$FPS" ]; then
    FPS=10
fi

python3 -m scripts.get_video "$IMAGE_DIR" "$OUTPUT_VIDEO" --fps "$FPS" 