#!/bin/bash

# Default config path
CONFIG_PATH="config/data_generator_config.yaml"

# Allow overriding config path via command line argument
if [ $# -eq 1 ]; then
    CONFIG_PATH=$1
fi

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

python3 -m scripts.data_generator_s3 "$CONFIG_PATH"