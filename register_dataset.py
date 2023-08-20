from detectron2.data.datasets import register_coco_instances

def register_my_dataset():
    #========= Register COCO dataset =========
    metadata = {"thing_classes": ["stalk", "spear"],
                "thing_colors": [(41,245,0), (200,6,6)]}
    # small test
    # register_coco_instances('asparagus_train_small', metadata, "/home/rayhuang/Asparagus_Dataset/COCO_Format/20230721_test/instances_train2017.json", "home/rayhuang/Asparagus_Dataset")
    # register_coco_instances('asparagus_val_small', metadata, "/home/rayhuang/Asparagus_Dataset/COCO_Format/20230721_test/instances_val2017.json", "home/rayhuang/Asparagus_Dataset")

    # full data
    register_coco_instances('asparagus_train_full_1920', metadata, "/home/rayhuang/Asparagus_Dataset/COCO_Format/20230817_Adam_1920/instances_train2017.json", "/home/rayhuang/Asparagus_Dataset")
    register_coco_instances('asparagus_val_full_1920', metadata, "/home/rayhuang/Asparagus_Dataset/COCO_Format/20230817_Adam_1920/instances_val2017.json", "/home/rayhuang/Asparagus_Dataset")
    register_coco_instances('asparagus_val_full', metadata, "/home/rayhuang/Asparagus_Dataset/COCO_Format/20230817_Adam_1920/instances_val2017.json", "home/rayhuang/Asparagus_Dataset")
