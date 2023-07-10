# exploring how one could use the pretrained models (with base64-encoded images as input) in an API

# %%
import os
import tensorflow as tf
import base64


# %%
def get_model_path(model_name, version=1):
    path = os.path.join(
        os.getcwd(), "models", model_name, str(version)
    )  # version number is required for things to work out-of-the-box with tensorflow serving
    return path


model_name = "resnet50_v1_fpn_640x640_base64"
model_path = get_model_path(model_name)
detector = tf.saved_model.load(model_path)
f = detector.signatures["serving_default"]
# %%
import numpy as np


def load_image(filename):
    with open(filename, "rb") as f:
        return np.array(f.read())


def get_images_in_folder(folder):
    return [
        os.path.join(folder, img) for img in os.listdir(folder) if img.endswith(".jpg")
    ]


def run_inference_on_image(image_path):
    base64_str = load_image(image_path)
    tensor = tf.convert_to_tensor([base64_str], dtype=tf.string)
    return f(bytes_inputs=tensor)


# %%

from tqdm import tqdm

results = []
for img in tqdm(get_images_in_folder("data/object-detection-SMALL")):
    results.append(run_inference_on_image(img))


# %%
def process_detection_result(result: dict):
    """
    Converts output of the object detection model to serializable format.
    """

    detection_boxes = result["detection_boxes"].numpy().astype(float).tolist()
    detection_classes = result["detection_classes"].numpy().astype(int).tolist()
    detection_scores = result["detection_scores"].numpy().astype(float).tolist()

    return {
        "detection_boxes": detection_boxes,
        "detection_classes": detection_classes,
        "detection_scores": detection_scores,
    }


# %%
processed_results = process_detection_result(results)


# %%
def simulate_upload_and_download(img_path):
    # Step 1: Load the image from disk
    image_data = load_image(img_path)

    # Step 2: Encode the image as base64 string
    base64_image = base64.b64encode(image_data).decode("utf-8")

    # Step 3: Transfer the base64 string (send it to the server)

    # Step 4: Decode the base64 string on the server
    decoded_image_data = base64.b64decode(base64_image)

    # Step 5: Load the image from the decoded bytes
    tensor = tf.convert_to_tensor([decoded_image_data], dtype=tf.string)
    result = f(bytes_inputs=tensor)
    return process_detection_result(result)


# %%
det_result = simulate_upload_and_download(
    "data/object-detection-SMALL/000000146462.jpg"
)
# %%
det_result
