## train

```bash
#!/bin/bash
current_time=$(date +"%Y%m%d_%H%M%S")

# Resnet50
CUDA_VISIBLE_DEVICES=1 python3 train_net.py --config-file configs/coco/instance-segmentation/Asparagus_config/exp_R50.yaml --num-gpus 1 MODEL.WEIGHTS model/maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth SOLVER.IMS_PER_BATCH 1 OUTPUT_DIR "output/${current_time}_R50"
```