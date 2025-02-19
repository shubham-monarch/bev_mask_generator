#!/bin/bash

# Default config path
CONFIG_PATH="config/eval-data-s3.yaml"

python3 -m scripts.eval-data-s3 --config "$CONFIG_PATH"