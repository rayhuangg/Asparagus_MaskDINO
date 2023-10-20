# Asparagus MaskDINO

## Scrip
### Train
```bash
#!/bin/bash
current_time=$(date +"%Y%m%d_%H%M%S")

# Resnet50
CUDA_VISIBLE_DEVICES=1 python3 train_net.py --config-file configs/coco/instance-segmentation/Asparagus_config/exp_R50.yaml --num-gpus 1 MODEL.WEIGHTS model/maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth SOLVER.IMS_PER_BATCH 1 OUTPUT_DIR "output/${current_time}_R50"
```


### Tensorboard
```bash
tensorboard --logdir output/ --port 1026
```

### 測試demo不同尺寸照片
```
datasets/Asparagus_Dataset/Adam_pseudo_label/202111_patrol/20211104_081521_.jpg 1920*1080
datasets/Asparagus_Dataset/Adam_pseudo_label/Justin_remain/390.jpg 3280*2464
datasets/Asparagus_Dataset/Adam_pseudo_label/Justin_remain/667.jpg 4032*3024
datasets/Asparagus_Dataset/Justin_labeled_data/162.jpg  4592*3448
datasets/Asparagus_Dataset/Justin_labeled_data/309.jpg  5472*3648
``````