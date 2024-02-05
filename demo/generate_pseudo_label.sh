# !/bin/bash

model_name="20231219_231336_swinL_FullData_1920"


# ============== Single folder demo ===================
input_path='/home/rayhuang/photo_demo_used/Pseudo/20240205try/20210903/20210903_13_57_57_.jpg'
output_path='/home/rayhuang/photo_demo_used/Pseudo/20240205try/20210903/'

CUDA_VISIBLE_DEVICES=0 python3 demo.py \
    --input "${input_path}" \
    --output "${output_path}/predict" \
    --json_output "${output_path}" \
    --config-file "../output/${model_name}/config.yaml" \
    --not_draw_bbox \
    --opts MODEL.WEIGHTS "../output/${model_name}/model_final.pth"


# ============== Multiple folder demo ===================

# Modify this line to decide input folder.
input_path='/home/rayhuang/photo_demo_used/Pseudo/20240205try/'

# Define an array of dates in the format YYYYMMDD
dates=($(find "${input_path}" -maxdepth 1 -type d -exec basename {} \; | grep -v "20240205try" | sort))
echo "Dates variable contains: ${dates[@]}"

# Loop through each date
# for date in "${dates[@]}"; do
#     # Set the output folder based on the date
#     output_path="${input_path}/${date}/predict/"
#     json_output_path="${input_path}/${date}/"

#     CUDA_VISIBLE_DEVICES=1 python3 demo.py \
#         --input "${input_path}${date}/*.jpg" \
#         --output "${output_path}" \
#         --json_output "${json_output_path}" \
#         --config-file "../output/${model_name}/config.yaml" \
#         --not_draw_bbox \
#         --opts MODEL.WEIGHTS "../output/${model_name}/model_final.pth"
# done