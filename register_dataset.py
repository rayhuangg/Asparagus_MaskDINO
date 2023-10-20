from detectron2.data.datasets import register_coco_instances

def register_my_dataset():
    #========= Register COCO dataset =========
    metadata = {"thing_classes": ["stalk", "spear"],
                "thing_colors": [(41,245,0), (200,6,6)]}
    metadata_4classes = {"thing_classes": ["stalk", "spear", "bar", "straw"],
                "thing_colors": [(41,245,0), (200,6,6), (230,217,51), (71,152,179)]}
    # small test
    # register_coco_instances('asparagus_train_small', metadata, "/home/rayhuang/Asparagus_Dataset/COCO_Format/20230721_test/instances_train2017.json", "home/rayhuang/Asparagus_Dataset")
    # register_coco_instances('asparagus_val_small', metadata, "/home/rayhuang/Asparagus_Dataset/COCO_Format/20230721_test/instances_val2017.json", "home/rayhuang/Asparagus_Dataset")

    # full data
    register_coco_instances('asparagus_train_full_1920', metadata, "/home/rayhuang/Asparagus_Dataset/COCO_Format/20230817_Adam_1920/instances_train2017.json", "/home/rayhuang/Asparagus_Dataset")
    # register_coco_instances('asparagus_val_full_1920', metadata, "/home/rayhuang/Asparagus_Dataset/COCO_Format/20230817_Adam_1920/instances_val2017.json", "/home/rayhuang/Asparagus_Dataset")
    # register_coco_instances('asparagus_val_full', metadata, "/home/rayhuang/Asparagus_Dataset/COCO_Format/20230817_Adam_1920/instances_val2017.json", "home/rayhuang/Asparagus_Dataset")
    register_coco_instances('asparagus_val_full_1920', metadata, "/home/rayhuang/Asparagus_Dataset/COCO_Format/20230817_Adam_1920/instances_val2017.json", "/home/rayhuang/Asparagus_Dataset")

    # adam "raw" dataset (the method hi register dataset)
    # register_coco_instances('asparagus_val', {'_background_': 0, 'stalk': 1, 'spear': 2}, "/home/rayhuang/Asparagus_Dataset/val/annotations.json", "/home/rayhuang/Asparagus_Dataset/val")

    # webserver used (with stalk, spear, straw, bar)
    register_coco_instances('asparagus_train_webserver', metadata_4classes, "/home/rayhuang/Asparagus_Dataset/COCO_Format/20231018_dataset_with_straw_bar/instances_train2017.json", "/home/rayhuang/Asparagus_Dataset")
    register_coco_instances('asparagus_val_webserver', metadata_4classes, "/home/rayhuang/Asparagus_Dataset/COCO_Format/20231018_dataset_with_straw_bar/instances_val2017.json", "/home/rayhuang/Asparagus_Dataset")
