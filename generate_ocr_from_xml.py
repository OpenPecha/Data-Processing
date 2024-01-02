import os
import cv2
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
from xml.dom import minidom
from natsort import natsorted

from Utils import rotate_image, create_dir, parse_labels


def generate_line_image_v1(
    image, contour, angle: float, kernel: tuple = (10, 16), iterations: int = 6
):
    image_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.drawContours(
        image_mask, [contour], contourIdx=-1, color=(255, 255, 255), thickness=-1
    )

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
    cropped_img = np.delete(cropped_img, np.where(~cropped_img.any(axis=0))[0], axis=1)
    return cropped_img


def preprocess_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(24, 24))
    bw = clahe.apply(image)
    #_, bw = cv2.threshold(bw, 170, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bw = cv2.adaptiveThreshold(bw, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 21)
    thresh_c = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)

    return thresh_c


def save_line_transcription(file_name, index, image, label, images_out, labels_out):
    line_img_path = os.path.join(images_out, f"{file_name}_{index}.jpg")
    cv2.imwrite(line_img_path, image)

    labe_file_path = os.path.join(labels_out, f"{file_name}_{index}.txt")

    with open(labe_file_path, "w", encoding="utf-8") as f:
        f.write(label)


def create_dataset(
    image_file,
    xml_file,
    images_out: str,
    labels_out: str,
    min_length: int,
    min_height: int,
    kernel_height: int,
    kernel_width: int,
    kernel_iterations: int,
    use_baseline: bool,
    binarize: bool
):
    file_name = os.path.basename(image_file).split(".")[0]
    image = cv2.imread(image_file)

    if binarize:
        image = preprocess_img(image, binarize)

    annotation_tree = minidom.parse(xml_file)

    text_areas = annotation_tree.getElementsByTagName("TextRegion")

    for text_area in text_areas:
        textlines = text_area.getElementsByTagName("TextLine")
        centers, contour_dict = parse_labels(textlines, y_offset=0)

        for line_idx, (_, (k, v)) in enumerate(zip(centers, contour_dict.items())):
            points, label, angle = v
            line_image = generate_line_image_v1(
                image,
                points,
                angle,
                kernel=(kernel_width, kernel_height),
                iterations=kernel_iterations,
            )

            if line_image.shape[0] != 0 or line_image.shape[1] != 0:
                if line_image.shape[1] > min_length and line_image.shape[0] > min_height:
                    save_line_transcription(
                        file_name, line_idx, line_image, label, images_out, labels_out
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-b", "--baseline", choices=["yes", "no"], required=False, default="no")
    parser.add_argument("--binarize", choices=["yes", "no"], required=False, default="no")
    parser.add_argument("--kernel_width", type=int, required=False, default=10)
    parser.add_argument("--kernel_height", type=int, required=False, default=16)
    parser.add_argument("--kernel_iterations", type=int, required=False, default=6)
    parser.add_argument("--min_length", type=int, required=False, default=300)
    parser.add_argument("--min_height", type=int, required=False, default=50)

    args = parser.parse_args()
    input_dir = args.input_dir#
    binarize = True if args.binarize == "yes" else False
    use_baseline = True if args.baseline == "yes" else False
    kernel_width = args.kernel_width
    kernel_height = args.kernel_height
    kernel_iterations = args.kernel_iterations
    min_length = args.min_length
    min_height = args.min_height

    dataset_out = os.path.join(input_dir, "Dataset")
    dataset_img_out = os.path.join(dataset_out, "lines")
    dataset_transcriptions_out = os.path.join(dataset_out, "transcriptions")

    create_dir(dataset_out)
    create_dir(dataset_img_out)
    create_dir(dataset_transcriptions_out)

    dataset_images = natsorted(glob(f"{input_dir}/*.jpg"))
    dataset_labels = natsorted(glob(f"{input_dir}/page/*.xml"))

    print(
        f"Volume: {input_dir} => Images: {len(dataset_images)} , Labels: {len(dataset_labels)}"
    )

    assert len(dataset_images) == len(dataset_labels)

    for image, annotation in tqdm(
        zip(dataset_images, dataset_labels), total=len(dataset_images)
    ):
        create_dataset(
            image,
            annotation,
            dataset_img_out,
            dataset_transcriptions_out,
            min_length,
            min_height,
            kernel_height,
            kernel_width,
            kernel_iterations,
            use_baseline,
            binarize
        )
