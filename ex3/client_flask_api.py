import base64
import io
import json
import os
import requests
from PIL import Image
import argparse
import time
import datetime


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


def write_json_to_file(json: str, path: str):
    with open(path, "w") as file:
        file.write(json)


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

    img_payload_dicts = [
        {"name": name, "content": content}
        for name, content in zip(filenames, encoded_images)
    ]

    url = "http://127.0.0.1:5000/api/detect"
    payload = json.dumps({"images": img_payload_dicts})
    headers = {"Content-Type": "application/json"}
    print(f"Sending data to API")

    upload_start_datetime = datetime.datetime.now()
    timestamp_filename = upload_start_datetime.strftime(
        "%Y-%m-%d_%H%M%S"
    )  # will be used in output filename to uniquely identify the result
    timestamp_str = upload_start_datetime.strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )  # for easy conversion to Python datetime https://stackoverflow.com/a/10805633/13727176
    response = requests.post(url, data=payload, headers=headers)
    response_received_time = time.time()
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
    result["request_sent_at"] = timestamp_str
    result["upload_time"] = upload_time
    result["input_folder_name"] = input_dir.split("/")[-1]
    result["api_url"] = url

    output_file_path = os.path.join(
        output_dir,
        f"result_{input_dir.split('/')[-1]}_{timestamp_filename}.json",
    )
    print(f"Writing result to '{output_file_path}'")

    os.makedirs(output_dir, exist_ok=True)
    write_json_to_file(
        json.dumps(result),
        output_file_path,
    )
