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


def decode_image(image):
    """
    Decode a base64 string to PIL image
    """
    return Image.open(io.BytesIO(base64.b64decode(image)))


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

    print(f"Found {len(img_paths)} images in {input_dir}")
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
    print(f"Uploading to the API")

    upload_start_time = time.time()
    response = requests.post(url, data=payload, headers=headers)
    response_received_time = time.time()
    upload_time = (
        response.elapsed.total_seconds()
    )  # https://stackoverflow.com/a/43260678/13727176
    print(f"Received response with status code {response.status_code}")
    total_request_time = response_received_time - upload_start_time
    server_response_time = total_request_time - upload_time

    print(f"Uploading took {upload_time} seconds")
    print(f"Response time was {server_response_time} seconds")
    print(f"Total request time was {total_request_time} seconds")

    timestamp_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    timestamp_str = datetime.datetime.now().strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )  # for easy conversion to Python datetime https://stackoverflow.com/a/10805633/13727176

    response_content = response.json()
    result = {}
    result["api_response"] = response_content
    result["upload_time"] = upload_time
    result["server_response_time"] = server_response_time
    result["total_request_time"] = total_request_time
    result["encoding_time"] = encoding_time
    result["timestamp"] = timestamp_str

    output_file_path = os.path.join(
        output_dir,
        f"result_{input_dir.split('/')[-1]}_{timestamp_filename}.json",
    )
    print(f"Writing result to {output_file_path}")

    os.makedirs(output_dir, exist_ok=True)
    write_json_to_file(
        json.dumps(result),
        output_file_path,
    )
