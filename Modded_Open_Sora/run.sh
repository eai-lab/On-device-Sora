#!/bin/bash


# CUDA_VISIBLE_DEVICES=3 python scripts/inference.py configs/opensora-v1-2/inference/test_config.py --prompt "zoom in video of a tram passing by city"
python scripts/inference.py configs/opensora-v1-2/inference/test_config.py "$@"
# python scripts/inference.py configs/opensora-v1-2/inference/test_config.py --prompt "A scene of a border collie running through a field in broad daylight"

# python freqmetric/gif_maker.py
