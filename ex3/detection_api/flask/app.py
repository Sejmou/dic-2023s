# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import io
import time

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from flask import Flask, request, jsonify, make_response
import datetime


def create_app():
    app = Flask(__name__)

    module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"  # SSD-based object detection model trained on Open Images V4 with ImageNet pre-trained MobileNet V2 as image feature extractor.
    detector = hub.load(module_handle).signatures["default"]
    app.detector = detector

    # routing http posts to this method
    @app.route("/api/detect", methods=["POST", "GET"])
    def main():
        processing_start_time = time.time()
        incoming_request_timestamp_str = datetime.datetime.now().strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )  # required by client for upload time calculation; using this format for easy conversion to Python datetime https://stackoverflow.com/a/10805633/13727176

        # get the json data from the request body and convert it to a python dictionary object
        data = request.get_json(force=True)

        filenames = [img["name"] for img in data["images"]]
        img_contents_base64 = [img["content"] for img in data["images"]]
        decoded_images = [decode_image(img) for img in img_contents_base64]

        data = detection_loop(app.detector, list(zip(filenames, decoded_images)))

        processing_time = time.time() - processing_start_time
        data["processing_time"] = processing_time
        data["request_received_at"] = incoming_request_timestamp_str

        return make_response(jsonify(data), 200)

    return app


def detection_loop(detector, images: list):
    """
    Performs object detection on a list of images.

    Args:
        detector: the object detection model
        images: list of tuples (filename, image) where image is a numpy ndarray of shape (width, height) if grayscale or (width, height, 3) if RGB
    """

    bounding_boxes = []
    inf_times = []

    for i, image in enumerate(images):
        filename, content = image
        print(f"Processing image {i + 1} of {len(images)} ({filename}))")

        img_tensor = convert_to_tensor(content)

        inference_start_time = time.time()

        result = detector(img_tensor)
        end_time = time.time()

        bounding_boxes.append(
            {
                "filename": filename,
                "boxes": process_detection_result(
                    result,
                    img_width=img_tensor.shape[1],
                    img_height=img_tensor.shape[2],
                ),
            }
        )

        inference_time = end_time - inference_start_time
        inf_times.append(inference_time)

    avg_inf_time = sum(inf_times) / len(inf_times)

    data = {
        "bounding_boxes": bounding_boxes,
        "inf_time": inf_times,
        "avg_inf_time": str(avg_inf_time),
    }

    return data


def decode_image(img):
    """
    Decodes a base64-encoded image.

    Args:
        img: base64-encoded image

    Returns:
        image: numpy ndarray of shape (width, height) if grayscale or (width, height, 3) if RGB
    """
    return np.array(Image.open(io.BytesIO(base64.b64decode(img))), dtype=np.float32)


def convert_to_tensor(image: np.ndarray):
    """
    Converts input image (numpy ndarray) to a tensor compatible with our chosen object detection model.

    Args:
        image: numpy ndarray of shape (width, height) if grayscale or (width, height, 3) if RGB

    Returns:
        image: tensor of shape (1, width, height, num_channels), as required by the model
    """
    image = tf.convert_to_tensor(image)

    if len(image.shape) == 2:
        # image is grayscale, convert to RGB
        image = tf.expand_dims(
            image, axis=-1
        )  # conversion function expects shape (width, height, 1)
        image = tf.image.grayscale_to_rgb(image)

    # convert to shape: [1, width, height, num_channels]
    image = tf.expand_dims(image, axis=0)
    return image


def process_detection_result(result: dict, img_width: int, img_height: int):
    """
    Converts output of object detection for our chosen object detection model (https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1)
    to a human-readable list of bounding boxes.

    Each bounding box is a dictionary with the following keys:
        - class_name: a string describing the class of the object detected
        - score: the confidence score of the detection
        - ymin: the y-coordinate of the top-left corner of the bounding box
        - xmin: the x-coordinate of the top-left corner of the bounding box
        - ymax: the y-coordinate of the bottom-right corner of the bounding box
        - xmax: the x-coordinate of the bottom-right corner of the bounding box
    """

    coords = result["detection_boxes"].numpy().astype(float)
    classes = result["detection_class_entities"].numpy().astype(str)
    scores = result["detection_scores"].numpy().astype(float)

    # convert the box coordinates from relative to absolute pixel values
    coords[:, 0] *= img_width
    coords[:, 1] *= img_height
    coords[:, 2] *= img_width
    coords[:, 3] *= img_height

    def get_box_dict(box, class_name, score):
        return {
            "class_name": class_name,
            "score": score,
            "ymin": box[0],
            "xmin": box[1],
            "ymax": box[2],
            "xmax": box[3],
        }

    # convert the bounding boxes to a list of dictionaries
    boxes = [
        get_box_dict(box, class_name, score)
        for box, class_name, score in zip(coords, classes, scores)
    ]

    return boxes


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0")
