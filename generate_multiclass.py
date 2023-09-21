"""
A simpe interface to generate multiclass masks from PageXML annotations:
use e.g.: python generate_multiclass.py --input_dir "PATH_TO_INPUT_DIRECTORY"

- optionally:
    -- annotate_lines: use TextLine Attribute instead of the annotated TextArea in order to visualize, viz. train text line detection
    --overlay: generates overlays of the mask and the image for reviewing purposes
"""


import os
import cv2
import logging
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from Utils import create_directory, sanity_check, generate_multi_mask


def generate_masks(
    directory: str, annotate_lines: str = "yes", overlay_preview: str = "no"
) -> None:
    """
    args:
    - overlay_preview: creates an overlay of the original image and the mask for debugging purposes
    - annotate lines: uses the "TextLine" Element in order to draw the line boxes instead of the TextArea
    """

    _images = natsorted(glob(f"{directory}/*.jpg"))
    _xml = natsorted(glob(f"{directory}/page/*.xml"))

    try:
        sanity_check(_images, _xml)
    except:
        logging.error(f"Image-Label Pairing broken in: {directory}")
        return

    mask_dir = os.path.join(directory, "Masks")
    output_dir = os.path.join(mask_dir, f"Multiclass")
    output_masks = os.path.join(output_dir, "Masks")

    create_directory(mask_dir)
    create_directory(output_dir)
    create_directory(output_masks)

    logging.info(f"created output directory: {output_dir}")

    for _img, _xml in tqdm(zip(_images, _xml), total=len(_images)):
        image_n = os.path.basename(_img).split(".")[0]
        img = cv2.imread(_img)
        clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(24, 24))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = clahe.apply(img)
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 11
        )
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        mask = generate_multi_mask(img, _xml, annotate_lines)

        mask_out = f"{output_masks}/{image_n}_mask.png"

        if overlay_preview == "yes":
            cv2.addWeighted(mask, 0.4, img, 1 - 0.4, 0, img)
            cv2.imwrite(mask_out, img)
        else:
            cv2.imwrite(mask_out, mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument(
        "--overlay", choices=["yes", "no"], required=False, default="no"
    )
    parser.add_argument(
        "--annotate_lines", choices=["yes", "no"], required=False, default="yes"
    )

    args = parser.parse_args()
    input_dir = args.input_dir
    annotate_lines = args.annotate_lines
    overlay = args.overlay

    generate_masks(input_dir, annotate_lines, overlay)
