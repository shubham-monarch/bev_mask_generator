#! /bin/bash

# Default value
URI="s3://occupancy-dataset/bev-05-cam-extrinsics/"

python3 -m scripts.get_leaf_cnt --uri="$URI" 