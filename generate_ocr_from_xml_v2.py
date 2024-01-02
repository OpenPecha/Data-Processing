import os
import cv2
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from xml.dom import minidom
from natsort import natsorted
from dataclasses import dataclass
from Utils import rotate_image, create_dir, group_lines, preprocess_img, parse_labels
from IPython.display import Image as ShowImage


@dataclass
class LineSample:
    image: np.array
    label: str
    x: float
    y: float
    width: float
    height: float


def generate_line_image_v1(image, contour, angle: float, kernel: tuple = (10, 16), iterations: int = 6):
    image_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.drawContours(image_mask, [contour], contourIdx=-1, color=(255, 255, 255), thickness=-1)

    dilate_k = np.ones(kernel, dtype=np.uint8)
    kernel_iterations = iterations

    image_mask = cv2.dilate(image_mask, dilate_k, iterations=kernel_iterations)
    image_masked = cv2.bitwise_and(image, image, mask=image_mask)

    if angle > 85.0 and angle != 90.0:
        angle = -(90 - angle)

    if angle == 90:
        angle = 0
        
    rotated_img = rotate_image(image_masked, angle=angle)

    cropped_img = np.delete(rotated_img, np.where(~rotated_img.any(axis=1))[0], axis=0)
    cropped_img = np.delete(cropped_img,np.where(~cropped_img.any(axis=0))[0], axis=1)

    return cropped_img

def blurr(image: np.array, blur_intensity: int = 4):
    bw = cv2.GaussianBlur(image, (blur_intensity, blur_intensity), 0)
    return bw

def adaptive_binarize(image: np.array, block_size: int = 13, c: float = 11, invert: bool = False):
    c = round(c, 2)
    
    bw = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)

    if invert:
        bw = cv2.bitwise_not(bw)

    return bw


def get_components(image: np.array):
    connectivity = 4 # or 8, check here: https://stackoverflow.com/questions/7088678/4-connected-vs-8-connected-in-connected-component-labeling-what-is-are-the-meri
    image = cv2.bitwise_not(image)
    output = cv2.connectedComponentsWithStats(image, 4, cv2.CV_32S)
    numLabels, labels, stats, centroids = output

    return numLabels, labels, stats, centroids

def get_component_info(index: int, stats, centroids):
           
    x = stats[index, cv2.CC_STAT_LEFT]
    y = stats[index, cv2.CC_STAT_TOP]
    w = stats[index, cv2.CC_STAT_WIDTH]
    h = stats[index, cv2.CC_STAT_HEIGHT]
    area = stats[index, cv2.CC_STAT_AREA]
    (cX, cY) = centroids[index]
    return x, y, w, h, cX, cY, area

def filter_components(image: np.array, y_offset: int = 3, area_threshold: int = 6, y_border: bool = True, x_border: bool = False, filter_area: bool = True):

    mask = np.zeros(image.shape, dtype="uint8")
    numLabels, labels, stats, centroids = get_components(image)

    for compnt_idx in range(0, numLabels):
        x, y, w, h, cX, cY, area = get_component_info(compnt_idx, stats, centroids)

        # component touches the border
        x_pos = x > 0 and x+w < image.shape[1] - 1
        y_pos = y > 0 and y+h < image.shape[0] - 1
        #y_pos = y > 0

        cy_filter = cY > y_offset and cY < image.shape[0] - y_offset
        area_size = area > area_threshold

        filters = []
        filters.append(cy_filter)
        
        if x_border:
            filters.append(x_pos)

        if y_border:
            filters.append(y_pos)

        if filter_area:
            filters.append(area_size)

        if all(filters):
            componentMask = (labels == compnt_idx).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)

    mask = cv2.bitwise_not(mask)
    return mask

def get_page_data(image: str, annotation: str) -> tuple[str, list[LineSample]]:
    
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = blurr(image, blur_intensity=3)
    image = cv2.dilate(image, (3,3), 3)
    image = adaptive_binarize(image, invert=True, block_size=11, c=11)

    annotation_tree = minidom.parse(annotation)
    textlines = annotation_tree.getElementsByTagName('TextLine')
    centers, contour_dict = parse_labels(textlines, y_offset=0)

    page_data: list[LineSample] = []

    for _, (_, (k, v)) in enumerate(zip(centers, contour_dict.items())):
        points, label, angle = v

        if len(label) > 30:
            (x, y), (width, height), angle = cv2.minAreaRect(points)
            line_image = generate_line_image_v1(image, points, angle, kernel=(6, 4), iterations=14)
            line_image = cv2.bitwise_not(line_image)
            y_off = int(line_image.shape[0] * 0.2)
            line_image = filter_components(line_image, y_offset=y_off, area_threshold=10, x_border=False)
            
            line_sample = LineSample(line_image, label, x, y, width, height)
            page_data.append(line_sample)

    return page_data


def save_line_transcription(file_name, index, image, label, images_out, labels_out):
    line_img_path = os.path.join(images_out, f"{file_name}_{index}.jpg")
    cv2.imwrite(line_img_path, image)

    labe_file_path = os.path.join(labels_out, f"{file_name}_{index}.txt")

    with open(labe_file_path, "w", encoding="utf-8") as f:
        f.write(label)



if __name__ == "__main__":
    volumes = ["W2KG229028-v1", "W2KG229028-v2", "W2KG229028-v3", "W2KG229028-v4", "W2KG229028-v5", "W2KG229028-v6", "W2KG229028-v7", "W2KG229028-v8", "W2KG229028-v9", "W2KG229028-v10", "W2KG229028-v14", "W2KG229028-v15","W2KG229028-v17", "W2KG229028-v20", "W2KG229028-v21", "W2KG229028-v26", "W2KG229028-v28", "W2KG229028-v30"]

    input_dir = "D:/Datasets/Tibetan/Glomanthang/Annotations_v2/Glomanthang-Annotations"
    dataset_out = os.path.join(input_dir, "Dataset_clean")
    dataset_img_out = os.path.join(dataset_out, "lines")
    dataset_transcriptions_out = os.path.join(dataset_out, "transcriptions")

    create_dir(dataset_out)
    create_dir(dataset_img_out)
    create_dir(dataset_transcriptions_out)

    for volume_dir in volumes:
        

        dataset_images = natsorted(glob(f"{input_dir}/{volume_dir}/*.jpg"))
        dataset_labels = natsorted(glob(f"{input_dir}/{volume_dir}/page/*.xml"))

        print(f"Volume: {volume_dir} => Images: {len(dataset_images)} , Labels: {len(dataset_labels)}")

        for image, annotation in tqdm(zip(dataset_images, dataset_labels), total=len(dataset_images)):
            file_name = os.path.basename(image).split(".")[0]
            page_data = get_page_data(image, annotation)

            for idx, line in enumerate(page_data):
                #print(line.label)
                save_line_transcription(file_name, idx, line.image, line.label, dataset_img_out, dataset_transcriptions_out)