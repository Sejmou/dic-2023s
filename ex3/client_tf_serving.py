"""
A client that performs inferences on an object detection model running on a server created with TensorFlow Serving using its REST API.
It uses images from the given input directory and sends them to the server.
The client processes the JSON response from the server (containing information about the detected objects).

The client expects a TensorFlow Serving ModelServer running a TensorFlow SavedModel for object detection trained on the
COCO 2017 dataset (https://cocodataset.org/), modified as described in https://cloud.google.com/blog/topics/developers-practitioners/add-preprocessing-functions-tensorflow-models-and-deploy-vertex-ai?hl=en
to make it accept JPEG a single JPEG image in base64 format as input.

This means that the output of the model (and hence the server) is the same as for example in this model: https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2

The script assumes that a model trained on the  is served.

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

MODEL_NAME = "ssd_mobilenet_v2_base64"  # the name under which the model is served on the TF Serving server
API_ENDPOINT_URL = f"http://localhost:8501/v1/models/{MODEL_NAME}:predict"

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


def load_images_from_folder(folder):
    return [Image.open(path) for path in get_image_paths(folder)]


def encode_image(image):
    """
    Encode a PIL image to base64 string
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def encode_images(images):
    return [encode_image(image) for image in images]


def create_api_payload(base64_img: str):
    return json.dumps({"instances": [{"bytes_inputs": {"b64": base64_img}}]})


def write_json_to_file(json: str, path: str):
    with open(path, "w") as file:
        file.write(json)


def get_prediction(base64_img: str):
    """
    Sends a POST request to the API endpoint with the given base64 image as payload.
    Processes the JSON response and returns a list of the detected bounding boxes.
    """
    payload = create_api_payload(base64_img)
    headers = {"Content-Type": "application/json"}
    response = requests.post(API_ENDPOINT_URL, data=payload, headers=headers)
    # reponse JSON is a list of dictionaries, each dictionary contains the predictions for one image
    # as the API only accepts one image per request, the list actually only contains a single dictionary
    predictions = response.json()["predictions"][0]
    boxes = process_prediction(predictions)
    return boxes


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Client for an object detection API built with TensorFlow Serving, accepting a base64-encoded image as input. Uploads images from a given input folder to the API and stores the bounding boxes of the detected objects received in the response."
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
        help="Path to directory where API response JSON should be stored.",
        default="data",
    )

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory '{input_dir}' does not exist.")

    if not os.path.isdir(output_dir):
        raise ValueError(f"Output directory '{output_dir}' does not exist.")

    img_paths = get_image_paths(input_dir)

    if len(img_paths) == 0:
        raise ValueError(
            f"Input directory {input_dir} does not contain any .jpg images."
        )

    print(f"Found {len(img_paths)} images in '{input_dir}'")
    print("Encoding images as array of Base64 strings")

    encoding_start_time = time.time()
    encoded_images = encode_images(load_images_from_folder(input_dir))
    encoding_time = time.time() - encoding_start_time
    print(f"Encoding took {encoding_time} seconds")

    filenames = [os.path.basename(path) for path in img_paths]

    boxes = {}
    for filename, img in zip(filenames, encoded_images):
        boxes[filename] = get_prediction(img)

    upload_start_datetime = datetime.datetime.now()
    timestamp_str = upload_start_datetime.strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )  # for easy conversion to Python datetime https://stackoverflow.com/a/10805633/13727176

    # TODO: figure out how to measure times and other metrics like in client_flask_api.py
    # response = requests.post(url, data=payload, headers=headers)
    # response_received_time = time.time()
    # request_time = (
    #     response.elapsed.total_seconds()
    # )  # https://stackoverflow.com/a/43260678/13727176
    # print(f"Received response with status code {response.status_code}")
    # print(
    #     f"Request (sending input data and receiving response with results) took {request_time} seconds"
    # )

    # response_content = response.json()

    # server_processing_time = response_content["processing_time"]
    # print(f"Server processed data in {server_processing_time} seconds")

    # data_transfer_time = request_time - server_processing_time
    # print(
    #     f"Data transfer between server and client (request (client -> server + response (server -> client)) took {data_transfer_time} seconds"
    # )

    # upload_received_datetime = datetime.datetime.strptime(
    #     response_content["request_received_at"], "%Y-%m-%dT%H:%M:%S.%fZ"
    # )
    # upload_time = (upload_received_datetime - upload_start_datetime).total_seconds()
    # print(f"Upload (sending request from client to server) took {upload_time} seconds")

    result = {}
    result["boxes"] = boxes
    # result["api_response"] = response_content
    # result["data_transfer_time"] = data_transfer_time
    # result["upload_time"] = upload_time
    # result["server_processing_time"] = server_processing_time
    # result["total_request_time"] = request_time
    # result["request_sent_at"] = timestamp_str
    result["requests_started_at"] = timestamp_str
    # result["upload_time"] = upload_time
    result["input_folder_name"] = input_dir.split("/")[-1]
    result["api_url"] = API_ENDPOINT_URL

    output_file_path = os.path.join(
        output_dir,
        f"result_{input_dir.split('/')[-1]}_{timestamp_str}.json",
    )
    print(f"Writing result to '{output_file_path}'")

    os.makedirs(output_dir, exist_ok=True)
    write_json_to_file(
        json.dumps(result),
        output_file_path,
    )
