import os
import cv2
import math
import json
import logging
import shutil
import numpy as np
import onnxruntime as ort
import xml.etree.ElementTree as etree

from typing import Optional
from xml.dom import minidom
from einops import rearrange
from datetime import datetime
from scipy.special import expit
from scipy.signal import find_peaks

from Config import COLOR_DICT


def get_model_info(config_file: str) -> tuple[str, int]:
    file = open(config_file)
    json_content = json.loads(file.read())
    onnx_model_file = json_content["model"]
    patch_size = json_content["patch_size"]

    return onnx_model_file, int(patch_size)


def get_file_name(x) -> str:
    return os.path.basename(x).split(".")[0]


def create_dir(dir_name: str) -> None:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_utc_time() -> str:
    t = datetime.now()
    s = t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    s = s.split(" ")

    return f"{s[0]}T{s[1]}"


def get_subdirs(root_dir: str) -> list[str]:
    data_root = root_dir
    sub_dirs = os.listdir(data_root)

    sub_dirs = [
        os.path.join(data_root, x)
        for x in sub_dirs
        if os.path.isdir(os.path.join(data_root, x))
    ]

    return sub_dirs


def create_directory(dir_path: str) -> None:
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except:
        logging.error("Failed creating output directories")


def remove_directory(dir_path: str) -> None:
    shutil.rmtree(dir_path)


def sanity_check(images: list[str], annotations: list[str]) -> None:
    image_names = [os.path.basename(x).split(".")[0] for x in images]
    annotation_names = [os.path.basename(x).split(".")[0] for x in annotations]

    for image_n, annotation_n in zip(image_names, annotation_names):
        assert image_n == annotation_n
        

def check_xml_status(xml_file: str):
    annotation_tree = minidom.parse(xml_file)
    doc_metadata = annotation_tree.getElementsByTagName("TranskribusMetadata")

    if len(doc_metadata) > 0:
        page_status = doc_metadata[0].attributes['status'].value

        if page_status == "DONE":
           return True
        
        else:
            return False

    else :
        return False

def read_labels(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [x.replace("\n", "") for x in lines]

        return lines


def resize_image(
    orig_img: np.array, target_width: int = 2048
) -> tuple[np.array, float]:
    if orig_img.shape[1] > orig_img.shape[0]:
        resize_factor = round(target_width / orig_img.shape[1], 2)
        target_height = int(orig_img.shape[0] * resize_factor)

        resized_img = cv2.resize(orig_img, (target_width, target_height))

    else:
        target_height = target_width
        resize_factor = round(target_width / orig_img.shape[0], 2)
        target_width = int(orig_img.shape[1] * resize_factor)
        resized_img = cv2.resize(orig_img, (target_width, target_height))

    return resized_img, resize_factor


def preprocess_img(image: np.array, blur_intensity: int = 7) -> np.array:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(24, 24))
    bw = clahe.apply(image)
    bw = cv2.GaussianBlur(bw, (blur_intensity, blur_intensity), 0)
    bw = cv2.adaptiveThreshold(
        bw, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 11
    )
    thresh_c = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    indices = np.where(thresh_c == 0)
    thresh_c = thresh_c.copy()
    thresh_c[indices[0], indices[1], :] = [0, 255, 0]

    return thresh_c


def get_color(key: str) -> list[int]:
    color = COLOR_DICT[key]
    color = color.split(",")
    color = [x.strip() for x in color]
    color = [int(x) for x in color]

    return color


def get_xml_point_list(attribute) -> np.array:
    """
    parses the PageXML Coords element and returns an np.array that conforms cv2 contours
    """
    coords = attribute.getElementsByTagName("Coords")
    base_points = coords[0].attributes["points"].value
    pts = base_points.split(" ")
    pts = [x for x in pts if x != ""]

    points = []
    for p in pts:
        x, y = p.split(",")
        a = int(float(x)), int(float(y))
        points.append(a)

    point_array = np.array(points, dtype=np.int32)

    return point_array


def get_paddings(img: np.array, patch_size: int = 256) -> tuple[int, int, int, int]:
    y_pad = (img.shape[0] - (img.shape[0] % patch_size) + patch_size) - img.shape[0]
    x_pad = (img.shape[1] - (img.shape[1] % patch_size) + patch_size) - img.shape[1]

    if y_pad % 2 != 0:
        y_pad1 = int(y_pad / 2)
        y_pad2 = int(y_pad / 2 + 1)

    else:
        y_pad1 = int(y_pad / 2)
        y_pad2 = int(y_pad / 2)

    if x_pad % 2 != 0:
        x_pad1 = int(x_pad / 2)
        x_pad2 = int(x_pad / 2 + 1)

    else:
        x_pad1 = int(x_pad / 2)
        x_pad2 = int(x_pad / 2)

    return y_pad1, y_pad2, x_pad1, x_pad2


def pad_image(
    img: np.array, patch_size: int = 64, is_mask=False
) -> tuple[np.array, tuple[float, float]]:
    x_pad = (math.ceil(img.shape[1] / patch_size) * patch_size) - img.shape[1]
    y_pad = (math.ceil(img.shape[0] / patch_size) * patch_size) - img.shape[0]

    if is_mask:
        pad_y = np.zeros(shape=(y_pad, img.shape[1], 3), dtype=np.uint8)
        pad_x = np.zeros(shape=(img.shape[0] + y_pad, x_pad, 3), dtype=np.uint8)
    else:
        pad_y = np.ones(shape=(y_pad, img.shape[1], 3), dtype=np.uint8)
        pad_x = np.ones(shape=(img.shape[0] + y_pad, x_pad, 3), dtype=np.uint8)
        pad_y *= 255
        pad_x *= 255

    img = np.vstack((img, pad_y))
    img = np.hstack((img, pad_x))

    return img, (x_pad, y_pad)


def patch_image(img: np.array, patch_size: int = 256, pad_value: int = 255) -> np.array:
    y_pad1, y_pad2, x_pad1, x_pad2 = get_paddings(img, patch_size=patch_size)
    padded_img = np.pad(
        img,
        pad_width=((y_pad1, y_pad2), (x_pad1, x_pad2), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )

    return rearrange(
        padded_img,
        "(h dim1) (w dim2) c -> (h w) (dim1 dim2 c)",
        dim1=patch_size,
        dim2=patch_size,
    )


def patch_image_v2(
    img: np.array, patch_size: int = 64, overlap: int = 2, is_mask=False
) -> tuple[list, int]:
    """
    A simple slicing function.
    Expects input_image.shape[0] and image.shape[1] % patch_size = 0
    """

    y_steps = img.shape[0] // patch_size
    x_steps = img.shape[1] // patch_size

    patches = []

    for y_step in range(0, y_steps):
        for x_step in range(0, x_steps):
            x_start = x_step * patch_size
            x_end = (x_step * patch_size) + patch_size

            crop_patch = img[
                y_step * patch_size : (y_step * patch_size) + patch_size, x_start:x_end
            ]
            patches.append(crop_patch)

    return patches, y_steps


def unpatch_image(image, pred_patches: list) -> np.array:
    patch_size = pred_patches[0].shape[1]

    x_step = math.ceil(image.shape[1] / patch_size)

    list_chunked = [
        pred_patches[i : i + x_step] for i in range(0, len(pred_patches), x_step)
    ]

    final_out = np.zeros(shape=(1, patch_size * x_step))

    for y_idx in range(0, len(list_chunked)):
        x_stack = list_chunked[y_idx][0]

        for x_idx in range(1, len(list_chunked[y_idx])):
            patch_stack = np.hstack((x_stack, list_chunked[y_idx][x_idx]))
            x_stack = patch_stack

        final_out = np.vstack((final_out, x_stack))

    final_out = final_out[1:, :]
    final_out *= 255

    return final_out


def unpatch_prediction(prediction: np.array, y_splits: int) -> np.array:
    prediction *= 255
    prediction_sliced = np.array_split(prediction, y_splits, axis=0)
    prediction_sliced = [np.concatenate(x, axis=1) for x in prediction_sliced]
    prediction_sliced = np.vstack(np.array(prediction_sliced))

    return prediction_sliced


def rotate_image(image: np.array, angle: float) -> np.array:
    rows, cols = image.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    
    return cv2.warpAffine(
        image, rot_matrix, (cols, rows), borderValue=(0, 0, 0)
    )


def rotate_page(
    original_image: np.array,
    line_mask: np.array,
    max_angle: float = 3.0,
    debug_angles: bool = False,
) -> tuple[np.array, np.array, float]:
    contours, _ = cv2.findContours(line_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # mask_threshold = (line_mask.shape[0] * line_mask.shape[1]) * 0.1
    # contours = [x for x in contours if cv2.contourArea(x) > mask_threshold]
    angles = [cv2.minAreaRect(x)[2] for x in contours]

    # angles = [x for x in angles if abs(x) != 0.0 and x != 90.0]
    low_angles = [x for x in angles if abs(x) != 0.0 and x < max_angle]
    high_angles = [
        x for x in angles if abs(x) != 90.0 and abs(x) != 180.0 and x > (90 - max_angle)
    ]

    if debug_angles:
        logging.info(angles)

    if len(low_angles) > len(high_angles) and len(low_angles) > 0:
        mean_angle = np.mean(low_angles)

    # check for clockwise rotation
    elif len(high_angles) > 0:
        mean_angle = -(90 - np.mean(high_angles))

    else:
        mean_angle = 0

    rows, cols = original_image.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), mean_angle, 1)
    rotated_img = cv2.warpAffine(
        original_image, rot_matrix, (cols, rows), borderValue=(0, 0, 0)
    )

    rotated_prediction = cv2.warpAffine(
        line_mask, rot_matrix, (cols, rows), borderValue=(0, 0, 0)
    )

    return rotated_img, rotated_prediction, mean_angle


"""
The code for the contour rotation is taken from: https://medium.com/analytics-vidhya/tutorial-how-to-scale-and-rotate-contours-in-opencv-using-python-f48be59c35a2
For this use case the contour is rotated 'back' using the center of the image, instead of the centroid of the contour
"""
def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

def rotate_contour(cnt, center, angle):
    cx = center[0]
    cy = center[1]

    cnt_norm = cnt - [cx, cy]

    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)

    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated



def generate_binary_mask(
    img: np.array, annotation_file: str, class_tag: str
) -> np.array:
    annotation_tree = minidom.parse(annotation_file)

    img_height = img.shape[0]
    img_width = img.shape[1]
    image_mask = np.zeros(shape=(int(img_height), int(img_width), 3), dtype=np.uint8)

    target_areas = annotation_tree.getElementsByTagName(class_tag)
    cv2.floodFill(
        image=image_mask, mask=None, seedPoint=(0, 0), newVal=get_color("background")
    )

    if len(target_areas) != 0:
        for target_area in target_areas:
            cv2.fillPoly(
                image_mask, [get_xml_point_list(target_area)], color=[255, 255, 255]
            )

    return image_mask


def generate_multi_mask(
    img: np.array, annotation_file: str, annotate_lines: str
) -> np.array:
    try:
        annotation_tree = minidom.parse(annotation_file)

    except:
        logging.error(f"Failed to parse: {annotation_file}")
        return

    img_height = img.shape[0]
    img_width = img.shape[1]
    image_mask = np.ones(shape=(int(img_height), int(img_width), 3), dtype=np.uint8)

    textareas = annotation_tree.getElementsByTagName("TextRegion")
    imageareas = annotation_tree.getElementsByTagName("ImageRegion")
    line_areas = annotation_tree.getElementsByTagName("TextLine")

    cv2.floodFill(
        image=image_mask, mask=None, seedPoint=(0, 0), newVal=get_color("background")
    )

    if len(textareas) != 0:
        for text_area in textareas:
            area_attrs = text_area.attributes["custom"].value

            if "marginalia" in area_attrs:
                cv2.fillPoly(
                    image_mask,
                    [get_xml_point_list(text_area)],
                    color=get_color("margin"),
                )
            
            # handles cases in which the annotators labelled images via a Textarea with "image" tag
            elif "image" in area_attrs:
                cv2.fillPoly(
                    image_mask,
                    [get_xml_point_list(text_area)],
                    color=get_color("image"),
                )

            elif "caption" in area_attrs:
                cv2.fillPoly(
                    image_mask,
                    [get_xml_point_list(text_area)],
                    color=get_color("caption"),
                )
            elif "page-number" in area_attrs:
                cv2.fillPoly(
                    image_mask,
                    [get_xml_point_list(text_area)],
                    color=get_color("margin"),
                )
            elif "footer" in area_attrs:
                cv2.fillPoly(
                    image_mask,
                    [get_xml_point_list(text_area)],
                    color=get_color("footer"),
                )
            elif "header" in area_attrs:
                cv2.fillPoly(
                    image_mask,
                    [get_xml_point_list(text_area)],
                    color=get_color("header"),
                )
            elif "table" in area_attrs:
                cv2.fillPoly(
                    image_mask,
                    [get_xml_point_list(text_area)],
                    color=get_color("table"),
                )

            else:
                if annotate_lines == "no":
                    cv2.fillPoly(
                        image_mask,
                        [get_xml_point_list(text_area)],
                        color=get_color("text"),
                    )

    if len(imageareas) != 0:
        for img in imageareas:
            cv2.fillPoly(
                image_mask, [get_xml_point_list(img)], color=get_color("image")
            )
    if annotate_lines == "yes":
        if len(line_areas) != 0:
            for line in line_areas:
                cv2.fillPoly(
                    image_mask, [get_xml_point_list(line)], color=get_color("line")
                )

    return image_mask


def get_lines(image: np.array, prediction: np.array) -> tuple[list[np.array], dict, tuple, list]:
    line_contours, _ = cv2.findContours(
        prediction, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    x, y, w, h = cv2.boundingRect(prediction)

    if len(line_contours) == 0:
        return None, None, None, None

    elif len(line_contours) == 1:
        bbox_center = (x + w // 2, y + h // 2)
        peaks = [bbox_center]
        sorted_contours = {bbox_center: line_contours[0]}
        line_images = get_line_images(image, sorted_contours)

        return line_images, sorted_contours, (x, y, w, h), peaks
    else:
        sorted_contours, peaks = sort_lines(prediction, line_contours)
        line_images = get_line_images(image, sorted_contours)

        return line_images, sorted_contours, (x, y, w, h), peaks


def group_lines(bbox_centers: list, line_threshold=20):
  sorted_bbox_centers = []
  tmp_line = []

  for i in range(0, len(bbox_centers)):
    # print(f"{i} -> {bbox_centers[i]}")
    
      if len(tmp_line) > 0:
        for s in range(0, len(tmp_line)):
          # is box on same line?
          y_diff = abs(tmp_line[s][1] - bbox_centers[i][1])

          # found new line?
          if y_diff > line_threshold:
            tmp_line.sort(key=lambda x:x[0])
            #print(f"Adding line: {tmp_line}")
            sorted_bbox_centers.append(tmp_line.copy())
            tmp_line.clear()
            #print(f"{i} -> cleared tmp_list: starting new tmp line: {bbox_centers[i]}")
            tmp_line.append(bbox_centers[i])
            break
          else:
          # print(f"{i} -> below thresh: appending to tmp line: {bbox_centers[i]}")
            tmp_line.append(bbox_centers[i])
            break
      else:
        #print(f"{i} -> tmp-zero: starting new tmp line: {bbox_centers[i]}")
        tmp_line.append(bbox_centers[i])

  sorted_bbox_centers.append(tmp_line)

  # sort each line by x-value
  for y in sorted_bbox_centers:
    y.sort(key=lambda x:x[0])

  return sorted_bbox_centers


def sort_lines(line_prediction: np.array, contours: tuple):
    """
    A preliminary approach to sort the found contours and sort them by reading lines. The relative distance between the lines is currently taken as roughly constant,
    wherefore mean // 2 is taken as threshold for line breaks. This might not work in scenarios in which the line distances are less constant.

    Args:
        - tuple of contours returned by cv2.findContours()
    Returns:
        - dictionary of {(bboxcenter_x, bbox_center_y) : [contour]}
        - peaks returned by find_peaks() marking the line breaks
    """

    horizontal_projection = np.sum(line_prediction, axis=1)
    horizontal_projection = horizontal_projection / 255
    mean = int(np.mean(horizontal_projection))
    peaks, _ = find_peaks(horizontal_projection, height=mean, width=4)

    # calculate the line distances
    line_distances = []
    for idx in range(len(peaks)):
        if idx < len(peaks) - 1:
            line_distances.append(
                peaks[(len(peaks) - 1) - idx] - (peaks[(len(peaks) - 1) - (idx + 1)])
            )

    if len(line_distances) == 0:
        line_distance = 0
    else:
        line_distance = int(
            np.mean(line_distances)
        )  # that might not work great if the line distances are varying a lot

    # get the bbox centers of each contour and keep a reference to the contour in contour_dict
    centers = []
    contour_dict = {}

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        y_center = y + (h // 2)
        x_center = x + (w // 2)
        centers.append((x_center, y_center))
        contour_dict[(x_center, y_center)] = contour

    centers = sorted(centers, key=lambda x: x[1])

    # associate bbox centers with the peaks (i.e. line breaks)
    cnt_dict = {}

    for center in centers:
        if center == centers[-1]:
            cnt_dict[center[1]] = [center]
            continue

        for peak in peaks:
            diff = abs(center[1] - peak)
            if diff <= line_distance // 2:
                if peak in cnt_dict.keys():
                    cnt_dict[peak].append(center)
                else:
                    cnt_dict[peak] = [center]

    # sort bbox centers for x value to get proper reading order
    for k, v in cnt_dict.items():
        if len(v) > 1:
            v = sorted(v)
            cnt_dict[k] = v

    # build final dictionary with correctly sorted bbox_centers by y and x -> contour
    sorted_contour_dict = {}
    for k, v in cnt_dict.items():
        for l in v:
            sorted_contour_dict[l] = contour_dict[l]

    return sorted_contour_dict, peaks


"""
The code for the contour rotation is taken from: https://medium.com/analytics-vidhya/tutorial-how-to-scale-and-rotate-contours-in-opencv-using-python-f48be59c35a2
For this use case the contour is rotated 'back' using the center of the image, instead of the centroid of the contour
"""


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def rotate_contour(cnt, center, angle):
    cx = center[0]
    cy = center[1]

    cnt_norm = cnt - [cx, cy]

    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)

    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated


def get_line_images(
    image: np.array,
    sorted_line_contours: dict,
    dilate_kernel: int = 20,
) -> list[np.array]:
    line_images = []

    for _, contour in sorted_line_contours.items():
        image_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.drawContours(
            image_mask, [contour], contourIdx=0, color=(255, 255, 255), thickness=-1
        )

        _, _, _, height = cv2.boundingRect(contour)

        if height < 60:
            dilate_iterations = dilate_kernel * 1
        elif height >= 60 and height < 90:
            dilate_iterations = dilate_kernel * 2
        else:
            dilate_iterations = dilate_kernel * 6

        dilated1 = cv2.dilate(
            image_mask,
            kernel=(dilate_kernel, dilate_kernel),
            iterations=dilate_iterations,
            borderValue=0,
            anchor=(-1, 0),
            borderType=cv2.BORDER_DEFAULT,
        )
        dilated2 = cv2.dilate(
            image_mask,
            kernel=(dilate_kernel, dilate_kernel),
            iterations=dilate_iterations,
            borderValue=0,
            anchor=(0, 1),
            borderType=cv2.BORDER_DEFAULT,
        )
        combined = cv2.add(dilated1, dilated2)
        image_masked = cv2.bitwise_and(image, image, mask=combined)

        cropped_img = np.delete(
            image_masked, np.where(~image_masked.any(axis=1))[0], axis=0
        )
        cropped_img = np.delete(
            cropped_img, np.where(~cropped_img.any(axis=0))[0], axis=1
        )

        line_images.append(cropped_img)

    return line_images


def generate_line_images(image: np.array, prediction: np.array):
    """
    Applies some rotation correction to the original image and creates the line images based on the predicted lines.
    """
    rotated_img, rotated_prediction, angle = rotate_page(
        original_image=image, line_mask=prediction
    )
    line_images, sorted_contours, bbox, peaks = get_lines(
        rotated_img, rotated_prediction
    )

    return line_images, sorted_contours, bbox, peaks, angle


def parse_labels(textlines: list, y_offset: int = -5):
    
    # TODO: use a descriptive dictionary or a struct to return the date for less cryptic usage down the road
    # calculate centers after the rotation of the contours
    
    centers = []
    contour_dict = {}

    for text_line_idx in range(len(textlines)):
        label = textlines[text_line_idx].getElementsByTagName("Unicode")

        if label[0].firstChild is not None:
            label = label[0].firstChild.nodeValue
        else:
            label = ""

        box_coords = textlines[text_line_idx].getElementsByTagName('Coords')
        img_box = box_coords[0].attributes['points'].value
        box_coordinates = img_box.split(' ')
        box_coordinates = [x for x in box_coordinates if x != ""]

        z = []
        for c in box_coordinates:
            x, y = c.split(",")
            a = [int(x), int(y)-y_offset]
            z.append(a)

        pts = np.array(z, dtype=np.int32)
        x,y,w,h = cv2.boundingRect(pts)
        min_area_rect = cv2.minAreaRect(pts)

        angle = min_area_rect[2]
        y_center = y + (h // 2)
        x_center = x + (w // 2)
        centers.append((x_center, y_center))
        contour_dict[(x_center, y_center)] = [pts, label, angle]
    
    return centers, contour_dict



def get_text_points(contour) -> str:
    points = ""
    for box in contour:
        point = f"{box[0][0]},{box[0][1]} "
        points += point
    return points


def get_text_line_block(coordinate: str, index: int, unicode_text: str):
    text_line = etree.Element(
        "Textline", id="", custom=f"readingOrder {{index:{index};}}"
    )
    text_line = etree.Element("TextLine")
    text_line_coords = coordinate

    text_line.attrib["id"] = f"line_9874_{str(index)}"
    text_line.attrib["custom"] = f"readingOrder {{index: {str(index)};}}"

    coords_points = etree.SubElement(text_line, "Coords")
    coords_points.attrib["points"] = text_line_coords
    text_equiv = etree.SubElement(text_line, "TextEquiv")
    unicode_field = etree.SubElement(text_equiv, "Unicode")
    unicode_field.text = unicode_text

    return text_line


def optimize_contour(cnt, e=0.001):
    epsilon = e * cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, epsilon, True)


def process_contours(
    # TODO: optionally use an area_threshold to filter small areas if they occur
    image: np.array,
    line_images: list[np.array],
    sorted_contours: dict,
    angle: float,
    area_threshold: float = 0.005,
) -> tuple[list, list]:
    line_contours = list(sorted_contours.values())
    line_contours = [optimize_contour(x) for x in line_contours]

    # back-rotate image contours to match the original input image
    image_height, image_width = image.shape[:2]
    line_contours = [
        rotate_contour(x, (image_width / 2, image_height / 2), angle)
        for x in line_contours
    ]

    filtered_images = []
    filtered_contours = []

    for line_image, contour in zip(line_images, line_contours):
        contour_area = cv2.contourArea(contour)

        if contour_area > (image.shape[0] * image.shape[1]) * 0.005:
            filtered_images.append(line_image)
            filtered_contours.append(contour)

    return filtered_images, filtered_contours


def build_xml_document(
    image: np.array,
    image_name: str,
    text_region_bbox,
    coordinates: list,
    text_lines: list,
) -> str:
    root = etree.Element("PcGts")
    root.attrib[
        "xmlns"
    ] = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    root.attrib["xmlns:xsi"] = "http://www.w3.org/2001/XMLSchema-instance"
    root.attrib[
        "xsi:schemaLocation"
    ] = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"

    metadata = etree.SubElement(root, "Metadata")
    creator = etree.SubElement(metadata, "Creator")
    creator.text = "Transkribus"
    created = etree.SubElement(metadata, "Created")
    created.text = get_utc_time()

    page = etree.SubElement(root, "Page")
    page.attrib["imageFilename"] = image_name
    page.attrib["imageWidth"] = f"{image.shape[1]}"
    page.attrib["imageHeight"] = f"{image.shape[0]}"

    reading_order = etree.SubElement(page, "ReadingOrder")
    ordered_group = etree.SubElement(reading_order, "OrderedGroup")
    ordered_group.attrib["id"] = f"1234_{0}"
    ordered_group.attrib["caption"] = "Regions reading order"

    region_ref_indexed = etree.SubElement(reading_order, "RegionRefIndexed")
    region_ref_indexed.attrib["index"] = "0"
    region_ref = "region_main"
    region_ref_indexed.attrib["regionRef"] = region_ref

    text_region = etree.SubElement(page, "TextRegion")
    text_region.attrib["id"] = region_ref
    text_region.attrib["custom"] = "readingOrder {index:0;}"

    text_region_coords = etree.SubElement(text_region, "Coords")
    text_region_coords.attrib["points"] = text_region_bbox

    if len(coordinates) == len(text_lines):
        for i in range(0, len(coordinates)):
            text_coords = get_text_points(coordinates[i])

            text_region.append(
                get_text_line_block(text_coords, i, unicode_text=text_lines[i])
            )
    else:
        for i in range(0, len(coordinates)):
            text_coords = get_text_points(coordinates[i])
            text_region.append(get_text_line_block(text_coords, i, unicode_text=""))

    xmlparse = minidom.parseString(etree.tostring(root))
    prettyxml = xmlparse.toprettyxml()

    return prettyxml


def predict_lines(
    image: np.array,
    inference_session: ort.InferenceSession,
    patch_size: int = 256,
    class_threshold: float = 0.8,
) -> np.array:
    resized_image, _ = resize_image(image)
    padded_img, (pad_x, pad_y) = pad_image(resized_image, patch_size)
    image_patches, _ = patch_image_v2(padded_img, patch_size)
    image_batch = np.array(image_patches)
    image_batch = image_batch.astype(np.float32)
    image_batch /= 255.0

    image_batch = np.transpose(image_batch, axes=[0, 3, 1, 2])  # make B x C x H xW

    ort_batch = ort.OrtValue.ortvalue_from_numpy(image_batch)
    ocr_results = inference_session.run_with_ort_values(
        ["output"], {"input": ort_batch}
    )
    prediction = ocr_results[0].numpy()
    prediction = np.squeeze(prediction, axis=1)
    prediction = expit(prediction)
    prediction = np.where(prediction > class_threshold, 1.0, 0.0)
    pred_list = [prediction[x, :, :] for x in range(prediction.shape[0])]

    unpatched_image = unpatch_image(resized_image, pred_list)

    cropped_image = unpatched_image[
        : unpatched_image.shape[0] - pad_y, : unpatched_image.shape[1] - pad_x
    ]

    # back_sized = cv2.resize(cropped_image, (int(cropped_image.shape[1] / resize_factor), int(cropped_image.shape[0] / resize_factor)))
    back_sized_image = cv2.resize(cropped_image, (image.shape[1], image.shape[0]))
    back_sized_image = back_sized_image.astype(np.uint8)

    return back_sized_image
