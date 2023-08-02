#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python demo.py \
    --config-file ../output/20230802_004113/config.yaml \
    --input ../datasets/Asparagus_Dataset/robot_regular_patrol/20211102/*.jpg --output demo_pic/ \
    --opts MODEL.WEIGHTS ../output/20230802_004113/model_0044999.pth