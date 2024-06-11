# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import math
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import csv
import tqdm
import pickle
import base64
import json
import numpy as np
import tqdm
import csv
from PIL import Image
from skimage import measure, morphology, feature
from pycocotools import mask
import matplotlib.pyplot as plt
from scipy import ndimage
from pathlib import Path

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances

from maskdino import add_maskdino_config
from predictor import VisualizationDemo
from register_dataset import register_my_datasets


# constants
WINDOW_NAME = "MaskDINO demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskdino demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--json_output",
        help="Directory for json result output."
        )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--csv_out_stalk_count",
        action='store_true',
        default=False,
        help="Determine whether to export the stalk count csv file",
    )
    parser.add_argument(
        "--spear_measure_csv_name",
        help="A filename indicate the spear real size mesurement."
        )
    parser.add_argument(
        "--output-filename",
        default=None,
        type=str,
        help="Determine the output extension filename",
    )
    parser.add_argument(
        "--not_draw_bbox",
        action="store_true",
        help="Determine whether the demo output need to draw the bbox box.",
    )
    parser.add_argument(
        "--draw_stalk_count_on_output",
        action="store_true",
        help="Determine whether the demo output need to draw the stalk count.",
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


def json_output(predictions, output_folder_name, img_filename, raw_image_path):
    '''
    Save predictions as json file
    '''
    out_filename = os.path.join(output_folder_name, os.path.splitext(img_filename)[0]) + '.json'
    with open(raw_image_path, 'rb') as img_file:
        img_data = base64.b64encode(img_file.read()).decode("utf-8")
    labels = {
        0: 'stalk',
        1: 'spear',
        2: 'bar',
        3: 'straw'
    }
    image_height, image_width = predictions['instances'].image_size
    # pred_boxes = np.asarray(predictions["instances"].pred_boxes)
    scores = predictions['instances'].scores.cpu().numpy()
    pred_classes = predictions['instances'].pred_classes.cpu().numpy()
    pred_masks = predictions["instances"].pred_masks.cpu().numpy()

    content = {
        "version": "4.5.5",
        "flags": {},
        "shapes": [],
        "imagePath": img_filename,
        "imageData": img_data,
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    for i in range(len(pred_classes)):
        # # clump type
        # if pred_classes[i] == 1:
        #     bbox = pred_boxes[i].cpu().numpy().tolist()
        #     shape ={
        #     "label": "clump",
        #     "points": [[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
        #     "group_id": i,
        #     "shape_type": "rectangle",
        #     "flags": {}
        #     }
        #     content["shapes"].append(shape)
        # else:

        segmentation = pred_masks[i]

        # plt.imsave(f"/home/rayhuang/photo_demo_used/temp/seg/segmentation_{i}_original.jpg", segmentation, cmap='gray')
        kernel = np.ones((3, 3), np.uint8)
        segmentation = cv2.dilate(segmentation, kernel, iterations=1)
        segmentation = cv2.erode(segmentation, kernel, iterations=1)

        # Add edge padding around the image to prevent the mask from sticking to the edge and causing separate recognition.
        padding_size = 10
        segmentation = cv2.copyMakeBorder(segmentation, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=0)
        segmentation = cv2.GaussianBlur(segmentation, (5, 5), 0)
        seg_contours = measure.find_contours(segmentation.T, 0.5)

        # from matplotlib.patches import Polygon
        # output_path = "/home/rayhuang/photo_demo_used/temp/seg/"
        # mask_path = os.path.join(output_path, f"segmentation_{i}_after.png")
        # plt.imsave(mask_path, segmentation, cmap='gray')

        # 儲存輪廓
        # plt.figure(figsize=(8, 8))
        # plt.imshow(segmentation, cmap='gray')
        # for seg in seg_contours:
        #     poly = Polygon(seg, edgecolor='r', fill=False)
        #     plt.gca().add_patch(poly)
        # plt.title(f"Contours - Instance {i}")
        # plt.axis('off')
        # contours_path = os.path.join(output_path, f"{i}_contours.png")
        # plt.savefig(contours_path)
        # plt.close()

        for seg in seg_contours:
            # Compensation coordinates
            seg[:, 0] -= padding_size
            seg[:, 1] -= padding_size

            # down sample
            if len(seg) > 30:
                sampled_seg = [seg[k].tolist() for k in range(0, len(seg), int(len(seg) / 30))]
            else:
                sampled_seg = [s.tolist() for s in seg]

            # Filter small instances
            if pred_classes[i] == 0: # stalk
                seg_area = measure.moments(seg)[0, 0]
                area_threshold = 800000 # Simple test decision, normal stalk is larger than 1000000
                if seg_area < area_threshold:
                    continue # skip this segment


            shape = {
                "label": labels[pred_classes[i]],
                "points": sampled_seg,
                "group_id": i,
                "shape_type": "polygon",
                "flags": {}
            }
            content["shapes"].append(shape)

    with open(out_filename, 'w+') as jsonfile:
        json.dump(content, jsonfile, indent=2)


def mask2skeleton(pred_mask):
    """
    Convert pred_masek into skeleton.

    Parameters:
        pred_mask(list): 2d list of boolean, Fasle for bg, True for fg.

    Returns:
        skeleton(list): 2d list of 0 and 1, o for bg, 1 for skeleton
    """
    pred_mask = np.asarray(pred_mask) + 0
    skeleton = morphology.skeletonize(pred_mask)
    return skeleton


# Justin ver.
def csv_out(predictions, filename, path):
    labels = {0: 'stalk', 1: 'spear', 2: 'bar', 3: 'straw'}
    image_height, image_width = predictions['instances'].image_size
    pred_boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
    scores = predictions['instances'].scores.cpu().numpy()
    pred_classes = predictions['instances'].pred_classes.cpu().numpy()
    pred_masks = predictions["instances"].pred_masks.cpu().numpy()
    table = []
    for i in range(len(pred_classes)):
        if pred_classes[i] == 1:
            boxs = pred_boxes[i].tolist()
            skeleton = mask2skeleton(pred_masks[i]).tolist()
            length_skeleton = np.count_nonzero(skeleton)
            length_box = boxs[3] - boxs[1]
            table.append([i, boxs[0], boxs[1], boxs[2], boxs[3], mask.encode(np.asfortranarray(pred_masks[i])), length_skeleton, length_box])
    table = sorted(table, key = lambda x: x[1])
    for i in range(len(table)):
        table[i][0] = i
    table = [['id', 'box(xmin)', 'box(ymin)', 'box(xmax)', 'box(ymax)', 'mask', 'length(skeleton)', 'length(box)']] + table
    with open(os.path.join(path, filename) + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(table)


def find_center(pred_boxes, num_boxes):
    """
    Finds the center coordinates of the bounding boxes.

    Args:
        pred_boxes (numpy.ndarray): Array containing bounding box coordinates.
        num_boxes (int): Number of bounding boxes(instanse).

    Returns:
        numpy.ndarray: Array of center coordinates.
    """
    center_coordinates = np.empty((0, 2), dtype=np.float64)
    for box_index in range(num_boxes):
        center = np.array([(pred_boxes[box_index][0] + pred_boxes[box_index][2]) / 2,
                           (pred_boxes[box_index][1] + pred_boxes[box_index][3]) / 2])
        center_coordinates = np.vstack((center_coordinates, center))
    return center_coordinates


def find_nearest_scale_index(spear_index, centers, scales_index):
    """
    Finds the index of the nearest scale to a given spear index.

    Args:
        spear_index (int): Index of the spear.
        centers (numpy.ndarray): Array of center coordinates.
        scales_index (numpy.ndarray): Array of indices of potential scales.

    Returns:
        int: Index of the nearest scale.
    """
    spear_center = centers[spear_index]
    distances = [np.linalg.norm(spear_center - centers[index]) for index in scales_index] # 2 norm
    nearest_scale_index = scales_index[np.argmin(distances)]
    return nearest_scale_index


def calculate_spear_real_dimensions(predictions, filename, image_size):
    """
    Calculates the real dimensions of spears based on the nearest scale.

    Args:
        predictions (dict): Prediction results from the model.
        filename (str): Name of the image file.
        image_size (tuple): Tuple representing the image dimensions (height, width).

    Returns:
        list: List of dictionaries containing spear measurements.
    """
    scale_lengths = {2: 150, 3: 200}  # Mapping of class indices to real-world lengths ('bar': 150, 'straw': 200)

    stalk_measurements = []
    pred_boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy()
    pred_classes = predictions["instances"].pred_classes.cpu().numpy()
    pred_masks = predictions["instances"].pred_masks.cpu().numpy()
    possible_scales_index = np.where(np.logical_or(pred_classes == 2, pred_classes == 3))[0]

    if len(possible_scales_index) == 0:
        print("No possible scales found.")
        return None

    centers = find_center(pred_boxes, len(pred_classes))
    for index in range(len(pred_boxes)):
        pred_class = pred_classes[index]
        if pred_class == 1:  # If it's spear
            nearest_scale_index = find_nearest_scale_index(index, centers, possible_scales_index)

            if nearest_scale_index not in possible_scales_index:
                print(f"Warning: No valid scale found. Skipping measurement.")
                continue

            # Scale real world measure part
            nearest_scale_class = pred_classes[nearest_scale_index]
            nearest_height, nearest_width = calculate_pixel_height_width(pred_masks, nearest_scale_index, image_size)
            scale_length = scale_lengths[nearest_scale_class]

            # Calculate spear's real height and width
            pixel_height, pixel_width = calculate_pixel_height_width(pred_masks, index, image_size)
            conversion_ratio = scale_length / nearest_height
            real_height = round((pixel_height * conversion_ratio), 2)
            real_width = round((pixel_width * conversion_ratio), 2)

            stalk_measurements.append({
                'filename': filename,
                'spear_index': index,
                'nearest_index': nearest_scale_index,
                'real_height': real_height,
                'real_width': real_width,
                'pixel_height': pixel_height,
                'pixel_width': round(pixel_width, 2),
            })

    return stalk_measurements


def calculate_pixel_height_width(pred_masks, target_index, image_size):
    """
    Calculates the height and width of the object based on the mask.

    Args:
        pred_masks (numpy.ndarray): Array of predicted masks.
        target_index (int): Index of the target object.
        image_size (tuple): Tuple representing the image dimensions (height, width).

    Returns:
        tuple: Height and width of the object.
    """
    props = measure.regionprops((pred_masks[target_index]).astype(np.uint8))

    # Extract major and minor axis lengths
    height = props[0].major_axis_length
    width = props[0].minor_axis_length

    # Calculate orientation angle, centroid, and rotate the mask
    angle = props[0].orientation * 180 / (math.pi)
    centroid = props[0].centroid
    mask_t_uint255 = (pred_masks[target_index].astype(np.uint8) * 255)
    M = cv2.getRotationMatrix2D(centroid, -angle, 1.0)
    rotated_img = cv2.warpAffine(mask_t_uint255, M, image_size)

    # Perform Canny edge detection on the rotated image
    edges = (feature.canny(rotated_img).astype(np.uint8) * 255)

    # Extract sub-images based on different points
    point_left = (int(centroid[0] - width), int(centroid[1] - height / 2))
    point_right = (int(centroid[0] + width), int(centroid[1] + height / 2))
    new_image_crop = rotated_img[point_left[1]:point_right[1], point_left[0]:point_right[0]]

    point_left = (int(centroid[0] - width / 2), int(centroid[1] - height / 2))
    point_right = (int(centroid[0] + width / 2), int(centroid[1] + height / 2))
    new_image_crop1 = rotated_img[point_left[1]:point_right[1], point_left[0]:point_right[0]]

    point_left = (int(centroid[0] - width), int(centroid[1] - height / 4))
    point_right = (int(centroid[0] + width), int(centroid[1] + height / 4))
    new_image_crop2 = edges[point_left[1]:point_right[1], point_left[0]:point_right[0]]

    # Calculate width based on Canny edge detection
    try:
        indices = np.where(new_image_crop2 != [0])
        indices_left = [indices[1][i] for i in range(len(indices[1])) if indices[1][i] < width]
        indices_right = [indices[1][i] for i in range(len(indices[1])) if indices[1][i] > width]
        canny_x_start = sum(indices_left) / len(indices_left)
        canny_x_end = sum(indices_right) / len(indices_right)
        width = abs(canny_x_end - canny_x_start)
    except:
        width = props[0].minor_axis_length

    # Skeletonize the mask to calculate height
    skeleton = (morphology.skeletonize(pred_masks[target_index].astype(np.uint8))).astype(np.uint8)
    height = int(np.sum(skeleton))

    return height, width


def dump_result_to_csv_file(results, csv_filename):
    # Fisrt time, create csv file
    if not os.path.isfile(csv_filename):
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'spear_index', 'nearest_index', 'real_height', 'real_width', 'pixel_height', 'pixel_width']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # Only appand result data
    with open(csv_filename, 'a+', newline='') as csvfile:
        fieldnames = ['filename', 'spear_index', 'nearest_index', 'real_height', 'real_width', 'pixel_height', 'pixel_width']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for result in results:
            writer.writerow(result)

if __name__ == "__main__":
    register_my_datasets()
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    # ColorMode: IMAGE, SEGMENTATION, IMAGE_BW
    # Choose one for predict label
    # SEGMENTATION: Use the label color define by the matadata.thing_colors in register dataset
    demo = VisualizationDemo(cfg, instance_mode=ColorMode.SEGMENTATION)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            filename = Path(path).name
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img, args.not_draw_bbox, args.draw_stalk_count_on_output)
            if hasattr(predictions["instances"], 'scores') and len(predictions["instances"].scores) > 0:
                len_of_prediction = len(predictions["instances"])
            else:
                len_of_prediction = 0

            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len_of_prediction)
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.spear_measure_csv_name:
                width, height, _ = img.shape
                results = calculate_spear_real_dimensions(predictions, filename, (height, width))
                if results is not None:
                    dump_result_to_csv_file(results, args.spear_measure_csv_name)

            if args.csv_out_stalk_count is True:
                pred_classes_list = predictions['instances'].pred_classes.tolist()
                count_of_stalk = pred_classes_list.count(0)
                with open('stalk_count.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if os.stat('stalk_count.csv').st_size == 0:  # Check if the file is empty
                        writer.writerow(["filename", "count_of_stalk"])  # Add headers only if the file is empty
                    writer.writerow([filename, count_of_stalk])

            if args.output:
                if os.path.isdir(args.output):
                    # assert os.path.isdir(args.output), args.output
                    if args.output_filename is not None:
                        base_name = os.path.splitext(os.path.basename(path))[0]  # Get the name of the file that does not include the expansion name
                        out_filename = os.path.join(args.output, f"{base_name}{args.output_filename}.jpg")
                    else:
                        out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    # assert len(args.input) == 1, "Please specify a directory with args.output"
                    os.makedirs(args.output)
                    if args.output_filename is not None:
                        base_name = os.path.splitext(os.path.basename(path))[0]  # Get the name of the file that does not include the expansion name
                        out_filename = os.path.join(args.output, f"{base_name}{args.output_filename}.jpg")
                    else:
                        out_filename = os.path.join(args.output, os.path.basename(path))
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit

            # Generate the pseudo label of input image
            if args.json_output:
                if not os.path.isdir(args.json_output):
                    os.makedirs(args.json_output)
                json_output_folder = args.json_output
                json_output(predictions, output_folder_name=json_output_folder,  img_filename=filename, raw_image_path=path)

    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
