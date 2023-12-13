import os
import cv2
import sys
import logging
import argparse
from tqdm import tqdm
from glob import glob
from natsort import natsorted
from Utils import get_file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--ext", type=str, required=False, default="tif")
    parser.add_argument("--cleanup", choices=["yes", "no"], required=False, default="no")

    args = parser.parse_args()

    input_dir = args.input_dir
    file_ext = args.ext
    cleanup = True if args.cleanup == "yes" else False

    images = natsorted(glob(f"{input_dir}/*.{file_ext}"))

    if len(images) == 0:
        logging.warning(f"No images found in the given directory: {input_dir}")
        sys.exit(1)

    for idx, image_path in tqdm(enumerate(images), total=(len(images))):
        img = cv2.imread(image_path)
        img_name = get_file_name(image_path)

        out_path = f"{input_dir}/{img_name}.jpg"
        cv2.imwrite(out_path, img)

        if cleanup:
            os.remove(image_path)
