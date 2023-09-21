""""
Using a line model for creating XML line predictions.

usage: python generate_xml_predictions.py --input_dir "data/0B7F9DBA/0B7F9DBA_1"

"""


import os
import cv2
import argparse
import onnxruntime as ort

from glob import glob
from tqdm import tqdm
from natsort import natsorted

from Utils import (
    create_dir,
    get_file_name,
    get_model_info,
    predict_lines,
    generate_line_images,
    process_contours,
    build_xml_document,
)

line_model_config = "models/line_model_config.json"
MODEL, PATCH_SIZE = get_model_info(line_model_config)
execution_providers = ["CPUExecutionProvider"]
inference_session = ort.InferenceSession(MODEL, providers=execution_providers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument(
        "--file_ext", choices=["jpg", "tif"], required=False, default="jpg"
    )

    args = parser.parse_args()
    input_dir = args.input_dir
    file_extension = args.file_ext

    images_paths = natsorted(glob(f"{input_dir}/*.{file_extension}"))
    xml_output = os.path.join(input_dir, "page")

    create_dir(xml_output)

    for idx, image_path in tqdm(enumerate(images_paths), total=len(images_paths)):
        image_name = get_file_name(image_path)
        image = cv2.imread(image_path)

        line_prediction = predict_lines(image, inference_session, PATCH_SIZE)
        line_images, sorted_contours, bbox, peaks, angle = generate_line_images(
            image, line_prediction
        )

        if line_images is not None:
            line_images, line_contours = process_contours(
                image, line_images, sorted_contours, angle
            )
            line_labels = []

            page_xml = build_xml_document(
                image, image_name, bbox, line_contours, line_labels
            )

            with open(f"{xml_output}/{image_name}.xml", "w", encoding="utf-8") as f:
                f.write(page_xml)

    print(f"Done!")
