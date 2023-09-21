"""
Using a line model for mapping line-based e-text to predicted lines, store them as individual files and/or PageXMl format.

usage: python generate_xml_predictions.py --input_dir "data/0B7F9DBA/0B7F9DBA_1"

optionally:
    --file_ext: specifiy the file extions for image formats, e.g. "tif", defaults to "jpg"
    --save_dataset: saves the predicted lines and the text line as individual files if you want them to use for OCR training, defaults to "yes"
    --save_xml: creates a PageXML file that can be opened in Transkribus, defaults to "yes"

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
    read_labels,
    get_model_info,
    build_xml_document,
    process_contours,
    generate_line_images,
    predict_lines,
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
    parser.add_argument(
        "--save_dataset", choices=["yes", "no"], required=False, default="yes"
    )
    parser.add_argument(
        "--save_xml", choices=["yes", "no"], required=False, default="yes"
    )

    args = parser.parse_args()
    input_dir = args.input_dir
    file_extension = args.file_ext
    save_dataset = args.save_dataset
    save_xml = args.save_xml

    images_paths = natsorted(glob(f"{input_dir}/*.{file_extension}"))
    labels_paths = natsorted(glob(f"{input_dir}/*.txt"))

    dataset_output = os.path.join(input_dir, "Dataset")
    lines_output = os.path.join(input_dir, "Dataset/lines")
    labels_output = os.path.join(input_dir, "Dataset/transcriptions")
    xml_output = os.path.join(input_dir, "page")

    create_dir(lines_output)
    create_dir(labels_output)
    create_dir(xml_output)

    mismatched_pages = []

    for image_path, label_path in tqdm(
        zip(images_paths, labels_paths), total=len(images_paths)
    ):
        image_name = get_file_name(image_path)

        image = cv2.imread(image_path)
        line_labels = read_labels(label_path)

        line_prediction = predict_lines(image, inference_session, PATCH_SIZE)
        line_images, sorted_contours, bbox, peaks, angle = generate_line_images(
            image, line_prediction
        )

        line_images, line_contours = process_contours(
            image, line_images, sorted_contours, angle
        )

        if len(line_images) == len(line_labels):
            # write line images and label files to disk
            if save_dataset == "yes":
                for idx, (line_image, line_label) in enumerate(
                    zip(line_images, line_labels)
                ):
                    out_image = f"{lines_output}/{image_name}_{idx}.jpg"
                    out_label = f"{labels_output}/{image_name}_{idx}.txt"

                    cv2.imwrite(out_image, line_image)

                    with open(out_label, "w", encoding="utf-8") as f:
                        f.write(line_label)

            # save PageXML
            if save_xml == "yes":
                page_xml = build_xml_document(
                    image, image_name, bbox, line_contours, line_labels
                )

                with open(f"{xml_output}/{image_name}.xml", "w", encoding="utf-8") as f:
                    f.write(page_xml)

        else:
            mismatched_pages.append(image_path)


        log_txt = f"{dataset_output}_mismatched.txt"

        with open(log_txt, "w", encoding="utf-8") as f:
            for log_path in mismatched_pages:
                f.write(f"{log_path}\n")

    print(f"Mismatched pages: {len(mismatched_pages)}. written log file to : {log_txt}")
    print(f"Done!")
