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

# initializing the flask app
app = Flask(__name__)


def detection_loop(images):
    # FasterRCNN+InceptionResNet V2: high accuracy, high inference time
    module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    # ssd+mobilenet V2: low accuracy, low inference time
    # module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"

    # load the model from the tensorflow hub
    detector = hub.load(module_handle).signatures['default']

    bounding_boxes = []
    inf_times = []
    upload_times = []

    for image in images:
        # get the current time before upload
        upload_start_time = time.time()

        # convert image to tensor and add batch dimension to it
        converted_img = tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...]

        # get the current time before inference
        inference_start_time = time.time()

        # perform inference on the image and get the result from the model output
        result = detector(converted_img)

        # get the current time after inference
        end_time = time.time()

        # append the result to the list of bounding boxes
        bounding_boxes.append(result['detection_boxes'].numpy())

        # append the inference time to the list of inference times
        inf_times.append(end_time - inference_start_time)

        # append the upload time to the list of upload times
        upload_times.append(end_time - upload_start_time)

    # calculate the average inference time and upload time
    avg_inf_time = sum(inf_times) / len(inf_times)

    # calculate the average upload time
    avg_upload_time = sum(upload_times) / len(upload_times)

    # convert the bounding boxes to a list of lists so that it can be serialized to json
    bounding_boxes = [box.tolist() for box in bounding_boxes]

    # create a dictionary of the data to be returned to the client
    data = {"bounding_boxes": bounding_boxes, "inf_time": inf_times, "avg_inf_time": str(avg_inf_time),
            "upload_time": upload_times, "avg_upload_time": str(avg_upload_time)}

    # return the response to the client as json with status code 200
    return make_response(jsonify(data), 200)


# routing http posts to this method
@app.route('/api/detect', methods=['POST', 'GET'])
def main():
    # get the json data from the request body and convert it to a python dictionary object
    data = request.get_json(force=True)

    # get the array of images from the json body
    imgs = data['images']

    images = []
    for img in imgs:
        # convert the base64 encoded image to a numpy array and append it to the list of images
        images.append((np.array(Image.open(io.BytesIO(base64.b64decode(img))), dtype=np.float32)))

    # call the detection loop method and return the response to the client
    return detection_loop(images)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
