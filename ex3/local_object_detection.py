# For running inference on the TF-Hub module.
import tensorflow as tf
import tensorflow_hub as hub

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import time
import argparse
import os
from tqdm import tqdm
import json


def draw_bounding_box_on_image(
    image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=()
):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (
        xmin * im_width,
        xmax * im_width,
        ymin * im_height,
        ymax * im_height,
    )
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
            fill=pick_font_color(color),
            font=font,
        )
        text_bottom -= text_height - 2 * margin


def pick_text_fill_color(bg_color):
    """
    Pick the color for the text fill based on the background color. If the background is sufficiently dark, pick white, else pick black.
    """
    bg_color = np.asarray(bg_color)
    bg_color = bg_color.astype(np.float32)
    bg_color = bg_color / 255.0
    bg_color = np.mean(bg_color)
    if bg_color < 0.5:
        return "white"
    else:
        return "black"


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())
    font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(
                class_names[i].decode("ascii"), int(100 * scores[i])
            )
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str],
            )
    return image_pil


def get_contrast_ratio(color1, color2):
    # Calculate the luminosity of a color
    def get_luminosity(color):
        r, g, b = color
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    # Calculate the contrast ratio
    luminosity1 = get_luminosity(color1)
    luminosity2 = get_luminosity(color2)
    ratio = (max(luminosity1, luminosity2) + 0.05) / (
        min(luminosity1, luminosity2) + 0.05
    )

    return ratio


def pick_font_color(bg_color):
    # Convert the color string to RGB tuple (if necessary)
    bg_rgb = ImageColor.getrgb(bg_color) if type(bg_color) == str else bg_color

    # Find the corresponding font color
    font_color = (
        "white" if get_contrast_ratio(bg_rgb, (255, 255, 255)) > 4.5 else "black"
    )

    return font_color


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def run_detector(detector, path):
    img = load_img(path)

    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    inference_time = end_time - start_time

    result = {key: value.numpy() for key, value in result.items()}

    pil_image_with_boxes = draw_boxes(
        img.numpy(),
        result["detection_boxes"],
        result["detection_class_entities"],
        result["detection_scores"],
    )

    return pil_image_with_boxes, inference_time


def store_as_json_file(dictionary, path):
    with open(path, "w") as fp:
        json.dump(dictionary, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs object detection model locally on images from provided directory. Draws labeled boxes for all detected objects onto each image and stores the output image in a 'local_results' subdirectory. A 'inference_stats.json' file is also created. Output images include formatted scores and label names for every bounding box."
    )
    parser.add_argument(
        "-i",
        "--img_dir",
        type=str,
        help="Path to directory where images are stored.",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Type of model to use. Options are 'small' or 'big'.",
        default="small",
    )
    small_model_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
    big_model_handle = (
        "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    )

    args = parser.parse_args()
    img_dir = args.img_dir
    model = args.model

    if model == "small":
        module_handle = small_model_handle
    elif model == "big":
        module_handle = big_model_handle
    else:
        raise ValueError("Model must be either 'small' or 'big'.")

    img_dir = args.img_dir

    if not os.path.exists(img_dir):
        raise ValueError("Image directory does not exist.")

    if not os.path.exists("local_results"):
        os.makedirs("local_results")

    img_filenames = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
    if len(img_filenames) == 0:
        raise ValueError("No images found in image directory.")
    print("Found %d images." % len(img_filenames))

    img_paths = [os.path.join(img_dir, f) for f in img_filenames]

    detector = hub.load(module_handle).signatures["default"]
    print("Model loaded.")

    out_dir = os.path.join(img_dir, "local_results", f"{model}_model")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    gpus = tf.config.list_physical_devices("GPU")
    gpu_available = len(gpus) > 0
    if gpu_available:
        print("GPU available.")
    else:
        print("GPU not available, running on CPU.")

    print(f"Running inference on images, writing results to '{out_dir}'...")
    inference_times = []
    for img_path in tqdm(img_paths):
        pil_image, inf_time = run_detector(detector, img_path)
        inference_times.append(inf_time)
        pil_image.save(os.path.join(out_dir, os.path.basename(img_path)), format="JPEG")

    stats = {
        "model": module_handle,
        "num_images": len(img_filenames),
        "avg_inference_time": np.mean(inference_times),
        "total_inference_time": np.sum(inference_times),
    }
    store_as_json_file(stats, os.path.join(out_dir, "inference_stats.json"))
