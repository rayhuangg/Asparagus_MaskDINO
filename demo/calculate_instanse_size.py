import math
import os
import csv
import cv2
import numpy as np

from skimage import measure, morphology, feature


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
            print(f"Create {csv_filename} file.")

    # Only appand result data
    with open(csv_filename, 'a+', newline='') as csvfile:
        fieldnames = ['filename', 'spear_index', 'nearest_index', 'real_height', 'real_width', 'pixel_height', 'pixel_width']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for result in results:
            writer.writerow(result)
