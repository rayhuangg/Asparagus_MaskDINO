#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python demo.py \
    --config-file ../output/20230802_004113/config.yaml \
    --input ../datasets/Asparagus_Dataset/Justin_labeled_data/309.jpg --output demo_pic/ \
    --opts MODEL.WEIGHTS ../output/20230802_004113/model_0044999.pth INPUT.MAX_SIZE_TEST 1920 INPUT.MIN_SIZE_TEST 1080


# datasets/Asparagus_Dataset/Justin_labeled_data/309.jpg  5472*3648
# datasets/Asparagus_Dataset/Adam_pseudo_label/202111_patrol/20211104_081521_.jpg 1920.1080