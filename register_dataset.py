import os
from detectron2.data.datasets import register_coco_instances

def register_dataset(d2_dataset_name, metadata, raw_folder_name):
    json_path = f"/home/rayhuang/Asparagus_Dataset/COCO_Format/{raw_folder_name}"
    dataset_path = "/home/rayhuang/Asparagus_Dataset"

    if os.path.exists(f"{json_path}/instances_train2017.json"):
        register_coco_instances(f"{d2_dataset_name}_train", metadata, f"{json_path}/instances_train2017.json", dataset_path)
    if os.path.exists(f"{json_path}/instances_val2017.json"):
        register_coco_instances(f"{d2_dataset_name}_val", metadata, f"{json_path}/instances_val2017.json", dataset_path)

def register_my_datasets():
    metadata_2classes = {"thing_classes": ["stalk", "spear"],
                         "thing_colors": [(41, 245, 0), (200, 6, 6)]}
    metadata_4classes = {"thing_classes": ["stalk", "spear", "bar", "straw"],
                         "thing_colors": [(41, 245, 0), (200, 6, 6), (230, 217, 51), (71, 152, 179)]}

    # small amount test
    register_dataset('asparagus_small', metadata_2classes, "20230803_test_small_dataset")

    # full data(Adam)
    register_dataset('asparagus_full_1920', metadata_2classes, "20230817_Adam_1920")
    register_coco_instances('asparagus_val_full', metadata_2classes, "/home/rayhuang/Asparagus_Dataset/COCO_Format/20230627_Adam_ver/instances_val2017.json", "/home/rayhuang/Asparagus_Dataset/")
    register_coco_instances('asparagus_val_full_1920', metadata_2classes, "/home/rayhuang/Asparagus_Dataset/COCO_Format/20230817_Adam_1920/instances_val2017.json", "/home/rayhuang/Asparagus_Dataset/")

    # webserver used (with stalk, spear, straw, bar)
    register_dataset('asparagus_webserver', metadata_4classes, "20231018_dataset_with_straw_bar")

    # adam "raw" dataset (the method Adam register(not sure metadata is useful or not))
    # register_coco_instances('asparagus_val', {'_background_': 0, 'stalk': 1, 'spear': 2}, "/home/rayhuang/Asparagus_Dataset/val/annotations.json", "/home/rayhuang/Asparagus_Dataset/val")

    # Test used
    register_dataset("20231213_ValidationSet_0point1", metadata_2classes, "20231213_ValidationSet_0point1")
    register_dataset("JoanAllData", metadata_2classes, "20240129_WithJoanAllData")
    ## full data(add 2021 pseudo label 6000 pics)
    register_dataset("Add2021pseudo", metadata_2classes, "20240208_Add2021PatrolData_6000pic")


