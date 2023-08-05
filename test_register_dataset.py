from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog



register_coco_instances('asparagus_train', {"thing_classes": ["stalk", "spear"], "thing_colors": [(255,255,0),(0,0,255)]}, "/home/rayhuang/MaskDINO/datasets/Asparagus_Dataset/COCO_Format/20230721_test/instances_train2017.json", "/home/rayhuang/MaskDINO/datasets/Asparagus_Dataset")
register_coco_instances('asparagus_val', {"thing_classes": ["stalk", "spear"], "thing_colors": {"stalk":(255,0,0), "spear":(0,255,250)}} , "/home/rayhuang/MaskDINO/datasets/Asparagus_Dataset/COCO_Format/20230721_test/instances_val2017.json", "/home/rayhuang/MaskDINO/datasets/Asparagus_Dataset")

print(f"{MetadataCatalog.get('asparagus_val')}")
print(f"{MetadataCatalog.get('asparagus_train')}")
