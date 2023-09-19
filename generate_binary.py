"""
A simpe interface to generate binary masks from PageXML annotations:
use e.g.: python generate_binary.py --input_dir "PATH_TO_INPUT_DIRECTORY"

- optionally:
    --target_class: a target XML-Element that should be used for the binary mask, defaults to "TextLine"
    --filter_blank: filters empty (black) tiles that contain to label information, defaults to "yes"
    --overlay: generates overlays of the mask and the image for reviewing purposes, defaults to "no"
"""

import os
import cv2
import logging
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from Utils import create_directory, sanity_check, remove_directory, generate_binary_mask



def generate_masks(
    directory: str,
    target_class: str = "TextLine",
    filter_blank : str = "yes",
    overlay_preview: str = "no"
) -> None:
    _images = natsorted(glob(f"{directory}/*.jpg"))
    _xml = natsorted(glob(f"{directory}/page/*.xml"))

    try:
        sanity_check(_images, _xml)
    except:
        logging.error(f"Image-Label Pairing broken in: {directory}")
        return
    
    mask_dir = os.path.join(directory, "Masks")
    create_directory(mask_dir)

    output_dir = os.path.join(mask_dir, f"Binary")
    
    if os.path.exists(output_dir):
        remove_directory(output_dir)

    create_directory(output_dir)
    
    for _img, _xml in tqdm(zip(_images, _xml), total=len(_images)):
        image_n = os.path.basename(_img).split(".")[0]
        img = cv2.imread(_img)
        mask = generate_binary_mask(img, _xml, target_class)
        
        if filter_blank == "yes":
            """
            skips all masks that are entirely black, i.e. containing no class information
            """
            if np.sum(mask) == 0:
                    continue
            else:
                mask_out = f"{output_dir}/{image_n}_mask.png"

                if overlay_preview == "yes":
                    cv2.addWeighted(mask, 0.4, img, 1 - 0.4, 0, img)
                    cv2.imwrite(mask_out, img)

                else:
                    cv2.imwrite(mask_out, mask)

        else:
            mask_out = f"{output_dir}/{image_n}_mask.png"

            if overlay_preview == "yes":
                cv2.addWeighted(mask, 0.4, img, 1 - 0.4, 0, img)
                cv2.imwrite(mask_out, img)

            else:
                cv2.imwrite(mask_out, mask)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--target_class", type=str, required=True, default="TextLine")
    parser.add_argument("--filter_blank", choices=["yes", "no"], required=False, default="yes")
    parser.add_argument("--overlay", choices=["yes", "no"], required=False, default="no")

    args = parser.parse_args()
    input_dir = args.input_dir
    target_class = args.target_class
    filter_blank = args.filter_blank
    overlay = args.overlay

    generate_masks(input_dir, target_class, filter_blank, overlay)