# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
A client that performs inferences on a object detection model running on a server created with TensorFlow Serving using its REST API.
The client downloads a test image of a cat, queries the serverwith the test image repeatedly and measures how long it takes to respond.

The client expects a TensorFlow Serving ModelServer running a TensorFlow SavedModel originally the one from:
https://github.com/tensorflow/models/tree/master/official/vision/image_classification/resnet#pretrained-models (link is dead lol)

Script adapted from https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/resnet_client.py
"""

from __future__ import print_function

import base64
import io
import json

import numpy as np
from PIL import Image
import requests

MODEL_NAME = "resnet50_v1_fpn_640x640"  # the name under which the model is served
SERVER_URL = f"http://localhost:8501/v1/models/{MODEL_NAME}:predict"
IMAGE_URL = "https://tensorflow.org/images/blogs/serving/cat.jpg"  # a cute cat image
MODEL_ACCEPT_JPG = False  # if True, the model accepts JPEG images in base64, otherwise it accepts tensors (says GitHub Copilot, not sure if that's what it actually means)
NORMALIZE_IMAGE = False  # if True, normalize the image pixel values to [0, 1] range
STORE_RESPONSE = True  # if True, store the response JSON in a file


def write_to_json(json: str, path: str):
    with open(path, "w") as file:
        file.write(json)


def main():
    # Download the image
    dl_request = requests.get(IMAGE_URL, stream=True)
    dl_request.raise_for_status()

    if MODEL_ACCEPT_JPG:
        # Compose a JSON Predict request (send JPEG image in base64).
        jpeg_bytes = base64.b64encode(dl_request.content).decode("utf-8")
        predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes
    else:
        # Compose a JOSN Predict request (send the image tensor).
        jpeg_rgb = Image.open(io.BytesIO(dl_request.content))
        img_arr = np.array(jpeg_rgb)
        if NORMALIZE_IMAGE:
            img_arr = img_arr / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)
        predict_request = json.dumps({"instances": img_arr.tolist()})

    # Send few requests to warm-up the model.
    for _ in range(3):
        response = requests.post(SERVER_URL, data=predict_request)
        response.raise_for_status()

    # Send few actual requests and report average latency.
    total_time = 0
    num_requests = 10
    for _ in range(num_requests):
        response = requests.post(SERVER_URL, data=predict_request)
        response.raise_for_status()
        total_time += response.elapsed.total_seconds()
        prediction = response.json()["predictions"][0]

    avg_latency = (total_time * 1000) / num_requests
    print(f"Average latency: {avg_latency} ms")

    # while the model is able to do multiple predictions at once, we only sent an image with a single object (cat)
    # let's check if the model actually detected a cat:
    # if it did, its detected bounding box should have the highest probability/score
    predicted_class_idx = np.argmax(prediction["detection_scores"])
    predicted_class = prediction["detection_classes"][predicted_class_idx]
    print(f"Predicted class: {predicted_class}")
    # assuming the model is trained on the COCO dataset, we can look up the class name for the class index here (class 17 is cat)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/

    if STORE_RESPONSE:
        write_to_json(response.json(), "response.json")


if __name__ == "__main__":
    main()
