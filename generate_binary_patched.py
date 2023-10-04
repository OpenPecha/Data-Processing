"""
A simpe interface to generate tiled binary masks from PageXML annotations:
use e.g.: python generate_binary_patched.py --input_dir "PATH_TO_INPUT_DIRECTORY"

- optionally:
    --target_class: a target XML-Element that should be used for the binary mask, defaults to "TextLine"
    --patch_size: specify the target tile size (e.g. 512), defaults to 256
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
from Utils import create_directory, sanity_check, generate_binary_mask, patch_image



def generate_masks(
    directory: str,
    target_class: str = "TextLine",
    filter_blank: bool = True,
    filter_threshold: str = True,
    threshold: float = 0.005,
    overlay_preview: bool = True
) -> None:
    _images = natsorted(glob(f"{directory}/*.jpg"))
    _xml = natsorted(glob(f"{directory}/page/*.xml"))

    try:
        sanity_check(_images, _xml)
    except:
        logging.error(f"Image-Label Pairing broken in: {directory}")
        return

    mask_dir = os.path.join(directory, "Masks")
    output_dir = os.path.join(mask_dir, f"Patched_Binary_{patch_size}")
    output_img_patches = os.path.join(output_dir, "Images")
    output_mask_patches = os.path.join(output_dir, "Masks")

    filter_threshold = True

    create_directory(mask_dir)
    create_directory(output_dir)
    create_directory(output_img_patches)
    create_directory(output_mask_patches)

    for _img, _xml in tqdm(zip(_images, _xml), total=len(_images)):
        image_n = os.path.basename(_img).split(".")[0]
        img = cv2.imread(_img)
        mask = generate_binary_mask(img, _xml, target_class)

        img_patches = patch_image(img, patch_size=patch_size)
        mask_patches = patch_image(mask, patch_size=patch_size, pad_value=0)

        for idx, (img_patch, mask_patch) in enumerate(zip(img_patches, mask_patches)):
            _img = img_patch.reshape(patch_size, patch_size, 3)
            _mask = mask_patch.reshape(patch_size, patch_size, 3)

            if filter_blank:
                """
                skips all masks that are entirely black, i.e. containing no class information
                """
                if np.sum(_mask) == 0:
                    continue

            if filter_threshold:
                tmp_mask = _mask / 255

                if np.sum(tmp_mask) > (mask.shape[0] * mask.shape[1]) * threshold:

                    img_out = f"{output_img_patches}/{image_n}_{idx}.jpg"
                    mask_out = f"{output_mask_patches}/{image_n}_{idx}_mask.png"

                    if overlay_preview:
                        cv2.addWeighted(_mask, 0.4, _img, 1 - 0.4, 0, _img)
                        cv2.imwrite(mask_out, _img)

                    else:
                        cv2.imwrite(img_out, _img)
                        cv2.imwrite(mask_out, _mask)

            else:
                img_out = f"{output_img_patches}/{image_n}_{idx}.jpg"
                mask_out = f"{output_mask_patches}/{image_n}_{idx}_mask.png"

                if overlay_preview:
                    cv2.addWeighted(_mask, 0.4, _img, 1 - 0.4, 0, _img)
                    cv2.imwrite(mask_out, _img)

                else:
                    cv2.imwrite(img_out, _img)
                    cv2.imwrite(mask_out, _mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--target_class", type=str, required=False, default="TextLine")
    parser.add_argument("--patch_size", required=False, type=int, default=256)
    parser.add_argument(
        "--filter_blank", choices=["yes", "no"], required=False, default="yes"
    )
    parser.add_argument(
        "--filter_threshold", choices=["yes", "no"], required=False, default="yes"
    )
    parser.add_argument(
        "--overlay", choices=["yes", "no"], required=False, default="no"
    )
    parser.add_argument("--threshold", type=float, required=False, default=0.005)

    args = parser.parse_args()
    input_dir = args.input_dir
    target_class = args.target_class
    patch_size = args.patch_size
    filter_blank = True if args.filter_blank == "yes" else False
    filter_threshold = True if args.filter_threshold == "yes" else False
    overlay = True if args.overlay == "yes" else False
    threshold = args.threshold

    generate_masks(input_dir, target_class, filter_blank, filter_threshold, threshold, overlay)
