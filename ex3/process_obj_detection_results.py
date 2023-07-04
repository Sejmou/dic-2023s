"""
Draws labeled boxes for all detected objects onto each image in the provided 'image_folder_path'.
Assumes all images were passed to the object detection API.
"""
import os
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import argparse
import json
from tqdm import tqdm


def draw_bounding_box_on_image(
    image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=()
):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=thickness,
        fill=color,
    )

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [
                (left, text_bottom - text_height - 2 * margin),
                (left + text_width, text_bottom),
            ],
            fill=color,
        )
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill="black",
            font=font,
        )
        text_bottom -= text_height - 2 * margin


def draw_detected_boxes(image_path: str, boxes: list, min_score: float):
    """
    Draws labeled boxes for all detected objects in the image. Assumption: the image was passed to the object detection API.

    Processes the 'boxes' part of the API response that contains an array of detected 'boxes' for the image.
    Each box is a dict with the following keys: 'ymin', 'xmin', 'ymax', 'xmax', 'class_name', 'score'.
    The coordinates are expected to be in range [0, 1].
    """
    colors = list(ImageColor.colormap.values())
    img = Image.open(image_path)
    img_width = img.width
    img_height = img.height
    font = ImageFont.load_default()
    for box in boxes:
        score = box["score"]
        if score >= min_score:
            ymin = box["ymin"] * img_height
            xmin = box["xmin"] * img_width
            ymax = box["ymax"] * img_height
            xmax = box["xmax"] * img_width
            class_name = box["class_name"]
            score = box["score"]
            display_str = "{}: {}%".format(class_name, int(100 * score))
            color = colors[hash(class_name) % len(colors)]
            draw_bounding_box_on_image(
                img,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str],
            )
    return img


def store_imgs_with_detected_boxes(
    image_folder_path, bounding_boxes, out_dir, min_score=None
):
    """
    Draws labeled boxes for all objects in the provided image_folder_path using the bounding_boxes dict, assuming all images were passed to the object detection API.
    Output images also include formatted scores and label names for every bounding box.

    Args:
        image_folder_path (str): Path to directory where images are stored.
        bounding_boxes (dict): Dictionary with the 'filename' of each processed images as key and its detected bounding boxes as the value.
        out_dir (str): Name of the subfolder where the processed images are stored.
        min_score (float, optional): Minimum confidence score for a bounding box to be drawn. Defaults to 0.1.

    Each processed image is stored in the specified subfolder with its original name.
    """
    print(
        f"Processing detection results for {len(bounding_boxes)} images in folder '{image_folder_path}'"
    )
    print(f"Storing images with detected boxes in '{image_folder_path}/{out_dir}'")

    min_score = min_score or 0.1

    filenames_and_boxes = list(bounding_boxes.items())

    for f_and_b in tqdm(filenames_and_boxes):
        filename, boxes = f_and_b
        print(f"Processing '{filename}'")
        image_path = os.path.join(image_folder_path, filename)
        print(f"Model detected {len(boxes)} objects in image.")
        img = draw_detected_boxes(image_path, boxes, min_score=min_score)
        img.save(os.path.join(out_dir, filename))


def load_json(path):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draws labeled boxes for all detected objects onto each image in the provided 'image_folder_path', assuming all images were passed to the object detection API. Processes the 'bounding_boxes' part of the API response that contains an array of objects with the 'filename' and detected 'boxes' for each image. Each image is stored in a subfolder of the provided 'image_folder_path', with the same name. Output images also include formatted scores and label names for every bounding box."
    )
    parser.add_argument(
        "-i",
        "--img_dir",
        type=str,
        help="Path to directory where images are stored.",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--result_file",
        type=str,
        help="Path to JSON file with result of object detection.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        help="Name of the subfolder where the processed images are stored.",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--min_score",
        type=float,
        help="Minimum confidence score for a bounding box to be drawn.",
    )
    args = parser.parse_args()
    img_dir = args.img_dir
    result_file = args.result_file
    min_score = args.min_score

    if not os.path.exists(img_dir):
        raise ValueError(f"Directory '{img_dir}' does not exist.")

    if not os.path.exists(result_file):
        raise ValueError(f"File '{result_file}' does not exist.")

    if not os.path.isdir(img_dir):
        raise ValueError(f"'{img_dir}' is not a directory.")

    if not os.path.isfile(result_file):
        raise ValueError(f"'{result_file}' is not a file.")

    if not result_file.endswith(".json"):
        raise ValueError(f"'{result_file}' is not a JSON file.")

    out_dir = os.path.join(img_dir, args.out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    result = load_json(result_file)
    bounding_boxes = result["boxes"]

    store_imgs_with_detected_boxes(
        img_dir, bounding_boxes, out_dir, min_score=min_score
    )
