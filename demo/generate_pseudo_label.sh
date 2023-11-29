# !/bin/bash

# Modify this line to decide output folder.
# output_path='/home/rayhuang/photo_demo_used/temp/'

# CUDA_VISIBLE_DEVICES=0 python demo.py \
#     --input '/home/rayhuang/photo_demo_used/Pseudo_label_straw/20211018/*.jpg' \
#     --output "${output_path}/predict" --json_output "${output_path}" \
#     --config-file ../output/20231018_134155_R50_webserver_used/config.yaml \
#     --opts MODEL.WEIGHTS ../output/20231018_134155_R50_webserver_used/model_final.pth



# Modify this line to decide input folder.
input_path='/home/rayhuang/photo_demo_used/1129_label_threshold/'

# Define an array of dates in the format YYYYMMDD
dates=($(find "${input_path}" -maxdepth 1 -type d -exec basename {} \; | grep -v "Pseudo_label" | sort))
echo "Dates variable contains: ${dates[@]}"

# Loop through each date
for date in "${dates[@]}"; do
    # Set the output folder based on the date
    output_path="/home/rayhuang/photo_demo_used/1129_label_threshold/${date}/predict/"
    json_output_path="/home/rayhuang/photo_demo_used/1129_label_threshold/${date}/"

    CUDA_VISIBLE_DEVICES=1 python3 demo.py \
        --input "${input_path}${date}/*.jpg" \
        --output "${output_path}" \
        --json_output "${json_output_path}" \
        --config-file ../output/20231018_134155_R50_webserver_used/config.yaml \
        --opts MODEL.WEIGHTS ../output/20231018_134155_R50_webserver_used/model_final.pth
done