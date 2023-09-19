import os
import cv2
import logging
import shutil
import numpy as np
from typing import List
from xml.dom import minidom
from einops import rearrange
from Config import COLOR_DICT

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


def sanity_check(images: List[str], annotations: List[str]) -> None:
    image_names = [os.path.basename(x).split(".")[0] for x in images]
    annotation_names = [os.path.basename(x).split(".")[0] for x in annotations]

    for image_n, annotation_n in zip(image_names, annotation_names):
        assert image_n == annotation_n


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


def patch_image(img: np.array, patch_size: int = 256, pad_value: int = 255) -> list:

    y_pad1, y_pad2, x_pad1, x_pad2 = get_paddings(img, patch_size=patch_size)
    padded_img = np.pad(img, pad_width=((y_pad1, y_pad2), (x_pad1, x_pad2), (0, 0)), mode="constant", constant_values=pad_value)

    return rearrange(
        padded_img, "(h dim1) (w dim2) c -> (h w) (dim1 dim2 c)", dim1=patch_size, dim2=patch_size
    )


def generate_binary_mask(
    img: np.array, annotation_file: str, class_tag: str
) -> np.array:
    try:
        annotation_tree = minidom.parse(annotation_file)

    except:
        logging.error(f"Failed to parse: {annotation_file}")

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


def generate_multi_mask(img: np.array, annotation_file: str, annotate_lines: str) -> np.array:
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
                    image_mask, [get_xml_point_list(text_area)], color=get_color("margin")
                )
            elif "caption" in area_attrs:
                cv2.fillPoly(
                    image_mask, [get_xml_point_list(text_area)], color=get_color("caption")
                )
            elif "page-number" in area_attrs:
                cv2.fillPoly(
                    image_mask, [get_xml_point_list(text_area)], color=get_color("margin")
                )
            elif "footer" in area_attrs:
                #print(f"Footer found! -> {annotation_file}")
                cv2.fillPoly(
                    image_mask, [get_xml_point_list(text_area)], color=get_color("footer")
                )
            elif "header" in area_attrs:
                #print(f"Header found! -> {annotation_file}")
                cv2.fillPoly(
                    image_mask, [get_xml_point_list(text_area)], color=get_color("header")
                )
            elif "table" in area_attrs:
                #print(f"Table found! -> {annotation_file}")
                cv2.fillPoly(
                    image_mask, [get_xml_point_list(text_area)], color=get_color("table")
                )

            else:
                if annotate_lines == "no":
                    cv2.fillPoly(
                        image_mask, [get_xml_point_list(text_area)], color=get_color("text")
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