import base64
import numpy as np
import json
import os
import requests
import argparse
import time
import datetime
from datetime import timezone


def get_image_paths(folder):
    """
    Returns a list of paths to all .jpg files in the given folder
    """
    return [
        os.path.join(folder, filename)
        for filename in os.listdir(folder)
        if filename.endswith(".jpg")
    ]


def load_image(filename):
    with open(filename, "rb") as f:
        return np.array(f.read())


def base64_encode_image(image):
    return base64.b64encode(image).decode("utf-8")


def process_img(img_path):
    """
    Encode a local image file as Base64 string that is ready to be sent to the API.
    """
    img_bytes = load_image(img_path)
    base64_image = base64_encode_image(img_bytes)
    return base64_image


def write_json_to_file(json: str, path: str):
    with open(path, "w") as file:
        file.write(json)


def get_current_timestamp():
    now = (
        datetime.datetime.utcnow()
    )  # using UTC time both on client and server for consistency
    timestamp_str = now.strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )  # for easy conversion to Python datetime https://stackoverflow.com/a/10805633/13727176
    return now, timestamp_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Client for the object detection API. Uploads images from a given input folder to the API and stores the detected objects received in the response."
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
    parser.add_argument(
        "-b",
        "--base-url",
        type=str,
        help="base URL of the API.",
        default="http://localhost:8502/",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="model to use for inference.",
        default="resnet50_v1_fpn_640x640",
    )

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    model = args.model
    base_url = args.base_url

    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory '{input_dir}' does not exist.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    img_paths = get_image_paths(input_dir)

    if len(img_paths) == 0:
        raise ValueError(
            f"Input directory {input_dir} does not contain any .jpg images."
        )

    print(f"Found {len(img_paths)} images in '{input_dir}'")
    print("Encoding images as array of Base64 strings")

    encoding_start_time = time.time()
    base64_imgs = [process_img(path) for path in img_paths]
    encoding_time = time.time() - encoding_start_time
    print(f"Encoding took {encoding_time} seconds")

    filenames = [os.path.basename(path) for path in img_paths]

    img_payload_dicts = [
        {"name": name, "content": content}
        for name, content in zip(filenames, base64_imgs)
    ]

    url = f"{base_url}/api/detect"
    payload = json.dumps({"images": img_payload_dicts, "model": model})
    headers = {"Content-Type": "application/json"}
    print(f"Sending data to API")

    upload_start_datetime, start_datetime_str = get_current_timestamp()
    response = requests.post(url, data=payload, headers=headers)
    request_time = (
        response.elapsed.total_seconds()
    )  # https://stackoverflow.com/a/43260678/13727176
    print(f"Received response with status code {response.status_code}")
    print(
        f"Request (sending input data and receiving response with results) took {request_time} seconds"
    )

    response_content = response.json()

    server_processing_time = response_content["processing_time"]
    print(f"Server processed data in {server_processing_time} seconds")

    data_transfer_time = request_time - server_processing_time
    print(
        f"Data transfer between server and client (request (client -> server + response (server -> client)) took {data_transfer_time} seconds"
    )

    upload_received_datetime = datetime.datetime.strptime(
        response_content["request_received_at"], "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    upload_time = (upload_received_datetime - upload_start_datetime).total_seconds()
    print(f"Upload (sending request from client to server) took {upload_time} seconds")

    result = {}
    result["api_response"] = response_content
    result["data_transfer_time"] = data_transfer_time
    result["upload_time"] = upload_time
    result["server_processing_time"] = server_processing_time
    result["total_request_time"] = request_time
    result["request_sent_at"] = start_datetime_str
    result["upload_time"] = upload_time
    result["input_folder_name"] = input_dir.split("/")[-1]
    result["api_url"] = url
    result["model"] = model

    output_file_path = os.path.join(
        output_dir,
        f"{'l' if 'localhost' in url else 'r'}_flask_{input_dir.split('/')[-1]}_{start_datetime_str}.json",
    )
    print(f"Writing result to '{output_file_path}'")
    write_json_to_file(
        json.dumps(result),
        output_file_path,
    )
