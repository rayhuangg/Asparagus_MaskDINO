# Asparagus MaskDINO

## Scrip
### Training script
```bash
#!/bin/bash
current_time=$(date +"%Y%m%d_%H%M%S")

# Resnet50
CUDA_VISIBLE_DEVICES=1 python3 train_net.py --config-file configs/coco/instance-segmentation/Asparagus_config/exp_R50.yaml --num-gpus 1 MODEL.WEIGHTS model/maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth SOLVER.IMS_PER_BATCH 1 OUTPUT_DIR "output/${current_time}_R50"
```

<details>
<summary>新增記錄功能training bash scrip</summary>

```bash
#!/bin/bash
# This script defines a function train_model, which facilitates training a neural network model with different configurations.
# It creates an output folder with a timestamp and provided name, backs up the training command, records it in 'train_history.txt', and executes the training.

# Usage:
# 1. Define the train_model function with parameters:
#    - $1: train_command (the training command)
#    - $2: output_folder_name (name for the output folder)
# 2. Call train_model with desired parameters.

# Example
# 1. Train a new model:
#     output_folder_name="R50_test"
#     train_command="python3 train_net.py --config-file xx --num-gpus 1 MODEL.WEIGHTS xx.pth SOLVER.IMS_PER_BATCH xx SOLVER.BASE_LR xx"
#     train_model "${train_command}" "${output_folder_name}"
# 2. RESUME training an existing model (Don't need to specify the output folder):
#     train_command="python3 train_net.py --resume --config-file xx --num-gpus 1 MODEL.WEIGHTS xx.pth "
#     train_model "${train_command}"

history_file="train_history.txt"

function train_model() {
    local current_time=$(date +"%Y%m%d_%H%M%S")
    local train_command="$1"
    local output_folder_name="$2"

    # Check if --resume is present in the train_command
    if [[ $train_command != *"--resume"* ]]; then
        # If --resume is not present, create a new OUTPUT folder
        local output_folder="output/${current_time}_${output_folder_name}"
        mkdir -p "${output_folder}"

        # Backup the training command to the OUTPUT folder
        local backup_script="${output_folder}/backup_train_command.sh"
        echo "${train_command} OUTPUT_DIR \"${output_folder}\"" > "${backup_script}"
        local train_command="${train_command} OUTPUT_DIR \"${output_folder}\""
    fi

    # Record the training command and execution time in train_history.txt
    echo "${current_time}: ${train_command}" >> "${history_file}"

    # # Execute the training
    eval "${train_command}"
}


# 20231220 152500
output_folder_name="new_train_scrip_test1"
train_command="CUDA_VISIBLE_DEVICES=1 python3 train_net.py --config-file configs/coco/instance-segmentation/Asparagus_config/exp_R50_normal.yaml --num-gpus 1 MODEL.WEIGHTS pretrain_model/maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth SOLVER.IMS_PER_BATCH 1 SOLVER.CHECKPOINT_PERIOD 50 SOLVER.MAX_ITER 300 SOLVER.BASE_LR 0.001"
train_model "${train_command}" "${output_folder_name}"

```

</details>

some useful opts arguments
```bash
current_time=$(date +"%Y%m%d_%H%M%S")
DATASETS.TRAIN '("asparagus_train_webserver",)'
OUTPUT_DIR "output/${current_time}_R50_WarmupCosineLR_resume_full_data"
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


即時更新nvidia-smi，-n 1設定更新頻率(秒)
```bash
watch -n 1 -d  nvidia-smi
```
