"""
A client that performs inferences on an object detection model running on a server created with TensorFlow Serving using its REST API.
It uses images from the given input directory and sends them to the server.
The client processes the JSON response from the server (containing information about the detected objects).

The client expects a TensorFlow Serving ModelServer running a TensorFlow SavedModel for object detection trained on the
COCO 2017 dataset (https://cocodataset.org/), modified as described in https://cloud.google.com/blog/topics/developers-practitioners/add-preprocessing-functions-tensorflow-models-and-deploy-vertex-ai?hl=en
to make it accept JPEG a single JPEG image in base64 format as input.

This means that the output of the model (and hence the server) is the same as for example in this model: https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2

The script assumes that a model trained on the COCO 2017 dataset (https://cocodataset.org/#download) is served.

Parts of the script were adapted from https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/resnet_client.py
"""
# %%
import base64
import io
import json
import os
import requests
from PIL import Image
import argparse
import time
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    coco_dataset_classes = pd.read_json(
        os.path.join("data", "coco2017_categories.json")
    ).set_index("id")
except:
    raise ValueError(
        "COCO 2017 dataset categories JSON file not found! Please download it using the download_coco2017_categories.py."
    )

# %%


def get_image_paths(folder):
    """
    Returns a list of paths to all .jpg files in the given folder
    """
    return [
        os.path.join(folder, filename)
        for filename in os.listdir(folder)
        if filename.endswith(".jpg")
    ]


def encode_image(image):
    """
    Encode a PIL image to base64 string
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def create_api_payload(base64_img: str):
    return json.dumps({"instances": [{"bytes_inputs": {"b64": base64_img}}]})


def write_json_to_file(json: str, path: str):
    with open(path, "w") as file:
        file.write(json)


def get_prediction(url: str, base64_img: str):
    """
    Sends a POST request to the API endpoint with the given base64 image as payload, outputting results.

    Args:
        url: the URL of the API endpoint
        base64_img: a base64-encoded JPEG image

    Returns:
        boxes: a list of bounding boxes of the detected objects in the image
        response_time: the time it took for the API to respond to the request
    """
    payload = create_api_payload(base64_img)
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=payload, headers=headers)
    # reponse JSON is a list of dictionaries, each dictionary contains the predictions for one image
    # as the API only accepts one image per request, the list actually only contains a single dictionary
    predictions = response.json()["predictions"][0]
    boxes = process_prediction(predictions)
    response_time = response.elapsed.total_seconds()
    return boxes, response_time


def process_prediction(prediction: dict, min_score: float = 0.0):
    """
    Converts output of an object detection model with same output dictionary as https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2
    to a human-readable list of bounding boxes. If min_score is passed, only bounding boxes with a confidence score of at least min_score are returned.

    Each bounding box is a dictionary with the following keys:
        - class_name: a string describing the class of the object detected
        - score: the confidence score of the detection
        - ymin: the y-coordinate of the top-left corner of the bounding box (value range: [0, 1])
        - xmin: the x-coordinate of the top-left corner of the bounding box (value range: [0, 1])
        - ymax: the y-coordinate of the bottom-right corner of the bounding box (value range: [0, 1])
        - xmax: the x-coordinate of the bottom-right corner of the bounding box (value range: [0, 1])
    """

    coords = prediction["detection_boxes"]
    class_idxs = np.array(prediction["detection_classes"]).astype(int)
    scores = prediction["detection_scores"]
    class_names = coco_dataset_classes.loc[class_idxs, "name"].values

    def get_box_dict(box, class_name, score):
        return {
            "class_name": class_name,
            "score": score,
            "ymin": box[0],
            "xmin": box[1],
            "ymax": box[2],
            "xmax": box[3],
        }

    # create list of dictionaries containing the detected bounding boxes
    boxes = [
        get_box_dict(box, class_name, score)
        for box, class_name, score in zip(coords, class_names, scores)
        if score >= min_score
    ]

    return boxes


def get_current_timestamp():
    now = datetime.datetime.now()
    timestamp_str = now.strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )  # for easy conversion to Python datetime https://stackoverflow.com/a/10805633/13727176
    return now, timestamp_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Client for an object detection API built with TensorFlow Serving, accepting a base64-encoded image as input. Uploads images from a given input folder to the API and stores the detection results + stats."
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        help="Path to directory where images should be uploaded from.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Path to directory where results should be stored.",
        default=os.path.join("data", "results"),
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Name of the model (API endpoint) to use for inference.",
        required=True,
    )
    parser.add_argument(
        "-b",
        "--base_url",
        type=str,
        help="Base URL of the API server.",
        default="http://localhost:8501",
    )

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    model = args.model
    base_url = args.base_url

    url = f"{base_url}/v1/models/{model}:predict"

    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory '{input_dir}' does not exist.")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    img_paths = get_image_paths(input_dir)

    if len(img_paths) == 0:
        raise ValueError(
            f"Input directory {input_dir} does not contain any .jpg images."
        )

    print(f"Found {len(img_paths)} images in '{input_dir}'")

    start_datetime, start_datetime_str = get_current_timestamp()

    # warm up the API with 3 dummy requests (experiments showed that first request is always slower)
    print("Warming up API with 3 dummy requests...")
    for _ in tqdm(list(range(3))):
        dummy_base64_img = encode_image(Image.open(img_paths[0]))
        get_prediction(url, dummy_base64_img)

    boxes_per_image = {}
    response_times = []
    print(
        "Processing images (one after another: loading, encoding to Base64, sending to API, storing result)"
    )
    with tqdm(total=len(img_paths)) as pbar:
        for path in img_paths:
            # Load image with Pillow and encode it as base64
            pil_img = Image.open(path)
            base64_img = encode_image(pil_img)

            # Send request to API and store results
            boxes, response_time = get_prediction(url, base64_img)

            filename = os.path.basename(path)
            boxes_per_image[filename] = boxes
            response_times.append(response_time)
            pbar.update(1)
        end_datetime, end_datetime_str = get_current_timestamp()

    average_response_time = np.mean(response_times)

    result = {}
    result["boxes"] = boxes
    result["started_at"] = start_datetime_str
    result["finished_at"] = end_datetime_str
    result["duration"] = (end_datetime - start_datetime).total_seconds()
    result["input_folder"] = input_dir.split("/")[-1]
    result["number_of_images"] = len(img_paths)
    result["response_times"] = response_times
    result["average_response_time"] = average_response_time
    result["api_url"] = url
    result["model"] = model
    # TODO: figure out if time spent processing on server can be measured somehow to differentiate between server and network latency
    # TODO: figure out how to measure time spent uploading each image to server

    output_file_path = os.path.join(
        output_dir,
        f"{'l' if 'localhost' in url else 'r'}_{input_dir.split('/')[-1]}({model})_{start_datetime_str}.json",
    )
    print(f"Writing result to '{output_file_path}'")
    write_json_to_file(
        json.dumps(result),
        output_file_path,
    )
