#!/bin/bash

output_path='/home/rayhuang/photo_demo_used/20231120 pseudo/20230930_part/'

CUDA_VISIBLE_DEVICES=0 python demo.py \
    --input '/home/rayhuang/photo_demo_used/20231120 pseudo/20230930_part/*.jpg' \
    --output "${output_path}predict/" --json_output "${output_path}json/" \
    --config-file ../output/20230812_110004_R50_normal/config.yaml \
    --opts MODEL.WEIGHTS ../output/20230812_110004_R50_normal/model_final.pth
