#! /bin/bash

# usage: diff_json.sh <source_json> <target_json>
python3 -m scripts.diff_json --s "index-s3/bev-dairy-18-02-25.json" --t "index-s3/bev-dairy-19-02-25.json" 