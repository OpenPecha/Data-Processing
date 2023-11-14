import os
import cv2
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
from xml.dom import minidom
from natsort import natsorted
from Utils import preprocess_img, group_lines


def generate_line_image_v2(idx: int, image: np.array, textlines, kernel_size: int = 4, iterations: int = 4):
    """
    :param idx:
    :param image:
    :param textlines:
    :param kernel_size:
    :param iterations:
    :return:
    """
    per_line_bboxes = []

    image_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    box_coords = textlines[idx].getElementsByTagName("Coords")
    img_box = box_coords[0].attributes["points"].value
    box_coordinates = img_box.split(" ")
    box_coordinates = [x for x in box_coordinates if x != ""]

    z = []
    for c in box_coordinates:
        x, y = c.split(",")
        a = [int(x), int(y)]
        z.append(a)

    pts = np.array(z, dtype=np.int32)
    per_line_bboxes.append(pts)

    cv2.drawContours(
        image_mask, [pts], contourIdx=-1, color=(255, 255, 255), thickness=-1
    )
    dilate_k = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    kernel_iterations = iterations

    image_mask = cv2.dilate(image_mask, dilate_k, iterations=kernel_iterations)
    image_masked = cv2.bitwise_and(image, image, mask=image_mask)

    cropped_img = np.delete(
        image_masked, np.where(~image_masked.any(axis=1))[0], axis=0
    )
    cropped_img = np.delete(cropped_img, np.where(~cropped_img.any(axis=0))[0], axis=1)

    indices = np.where(cropped_img[:, :, 1] == 0)
    clear = cropped_img.copy()
    clear[indices[0], indices[1], :] = [255, 255, 255]
    clear_bw = cv2.cvtColor(clear, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(clear_bw, 170, 255, cv2.THRESH_BINARY)

    text_content = textlines[idx].getElementsByTagName("Unicode")

    return thresh, text_content


def save_line_transcription(
    file_name, index, image, transcription, img_outpath, lbl_out_path
):
    line_file = os.path.join(lbl_out_path, f"{file_name}_{index}.txt")

    if not transcription[0].firstChild is None:
        with open(line_file, "w", encoding="utf-8") as f:
            f.write(transcription[0].firstChild.nodeValue)

        target_img_file = os.path.join(img_outpath, f"{file_name}_{index}.jpg")
        cv2.imwrite(target_img_file, image)


def create_mask_from_annotations(
    image_path: str, xml_file: str, img_outpath: str, lbl_outpath: str, group_lines: bool = True, check_xml_status: bool = True
):

    annotation_tree = minidom.parse(xml_file)
    doc_metadata = annotation_tree.getElementsByTagName("TranskribusMetadata")

    if check_xml_status:
        if len(doc_metadata) > 0:
            page_status = doc_metadata[0].attributes['status'].value

            if page_status == "DONE":
                # print(f"Found {len(textlines)} Lines")
                file_name = os.path.basename(image_path).split(".")[0]
                image = cv2.imread(image_path)
                image = preprocess_img(image)
                textlines = annotation_tree.getElementsByTagName("TextLine")

                for idx in range(len(textlines)):
                    line_img, text_content = generate_line_image_v2(idx, image, textlines)
                    save_line_transcription(
                        file_name, idx, line_img, text_content, img_outpath, lbl_outpath
                        )

    else:
        file_name = os.path.basename(image_path).split(".")[0]
        image = cv2.imread(image_path)
        image = preprocess_img(image)
        textlines = annotation_tree.getElementsByTagName("TextLine")

        for idx in range(len(textlines)):
            line_img, text_content = generate_line_image_v2(idx, image, textlines)
            save_line_transcription(
                file_name, idx, line_img, text_content, img_outpath, lbl_outpath
            )

if __name__ == "__main__":
    """
    e.g. python generate_dataset.py --input_dir "J:\Datasets\Tibetan\OCR\GloManThang\Volumes_v2" --use_subdirs "yes"
    e.g. python generate_dataset.py --input_dir "J:\Datasets\Tibetan\OCR\Karmapa8\XML" --use_subdirs "no"
     """

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--use_subdirs", choices=["yes", "no"], required=False, default="yes")
    parser.add_argument("--kernel", type=str, required=False, default="3")
    parser.add_argument("--kernel_iterations", type=str, required=False, default="4")
    args = parser.parse_args()

    ds_root = args.input_dir

    dataset_out = os.path.join(ds_root, "OCRDataset")
    img_out = os.path.join(dataset_out, "lines")
    label_out = os.path.join(dataset_out, "transcriptions")

    if not os.path.exists(dataset_out):
        os.makedirs(dataset_out)

    if not os.path.exists(img_out):
        os.makedirs(img_out)

    if not os.path.exists(label_out):
        os.makedirs(label_out)

    if args.use_subdirs and args.use_subdirs == "no":
        images = natsorted(glob(f"{ds_root}/*.jpg"))
        labels = natsorted(glob(f"{ds_root}/page/*.xml"))

        print(f"Volume {ds_root} => Images: {len(images)} - XML-Files: {len(labels)}")
        print("generating dataset....")
        for image, label in tqdm(zip(images, labels), total=len(images)):
            create_mask_from_annotations(image, label, img_out, label_out, check_xml_status=True)

    else:
        for sub_dir in Path(ds_root).iterdir():
            print(f"sub dir: {sub_dir}")
            images = natsorted(glob(f"{sub_dir}/*.jpg"))
            labels = natsorted(glob(f"{sub_dir}/page/*.xml"))
            #images = natsorted(glob(f"{sub_dir}/*.jpg"))
            #labels = natsorted(glob(f"{sub_dir}/page/*.xml"))

            print(f"Volume {sub_dir} => Images: {len(images)} - XML-Files: {len(labels)}")

            for image, label in tqdm(zip(images, labels), total=len(images)):
                create_mask_from_annotations(image, label, img_out, label_out, check_xml_status=False)