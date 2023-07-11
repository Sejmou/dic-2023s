import time
import os
import tensorflow as tf
from flask import Flask, request, jsonify, make_response
import datetime
from datetime import timezone
import base64


def create_app():
    app = Flask(__name__)

    model_config = {
        # key is the model name, value is the path to the saved model in the file system
        "resnet50_v1_fpn_640x640": get_model_path("resnet50_v1_fpn_640x640_base64"),
        "ssd_mobilenet_v2": get_model_path("ssd_mobilenet_v2_base64"),
    }

    models = {}
    for model_name, model_path in model_config.items():
        print(f"Loading model {model_name} from {model_path}")
        detector = tf.saved_model.load(model_path)
        print(f"Model {model_name} loaded successfully")
        predict_fn = detector.signatures["serving_default"]
        models[model_name] = predict_fn

    app.models = models

    # routing http posts to this method
    @app.route("/api/detect", methods=["POST"])
    def main():
        (
            incoming_request_time,
            incoming_request_time_str,
        ) = get_current_timestamp()  # required by client for upload time calculation

        # get the json data from the request body and convert it to a python dictionary object
        data = request.get_json(force=True)

        model = data.get("model")
        if not model:
            return make_response(
                jsonify(
                    {
                        "error": f'Model not specified. Please add it to the payload (key: "model"). Options are {model_config.keys()}'
                    }
                ),
                400,
            )
        predict_fn = app.models.get(model)
        if not predict_fn:
            return make_response(jsonify({"error": f"Model {model} not found"}), 404)
        filenames = [img["name"] for img in data["images"]]
        base64_imgs = [img["content"] for img in data["images"]]
        img_bytes = [decode_base64(img) for img in base64_imgs]

        data = detection_loop(predict_fn, list(zip(filenames, img_bytes)))

        processing_time = (
            datetime.datetime.utcnow() - incoming_request_time
        ).total_seconds()
        data["processing_time"] = processing_time
        data["request_received_at"] = incoming_request_time_str

        return make_response(jsonify(data), 200)

    return app


def get_model_path(model_name, version=1):
    path = os.path.join(
        os.getcwd(), "models", model_name, str(version)
    )  # version number is required for things to work out-of-the-box with tensorflow serving
    return path


def decode_base64(string):
    return base64.b64decode(string)


def detection_loop(predict_fn, images: list):
    """
    Performs object detection on a list of images.

    Args:
        predict_fn: the prediction function of the object detection model (accepting Base64-encoded image as input)
        images: list of tuples (filename, img_bytes) where filename is the name of the image and img_bytes contains the Base64-encoded image
    """

    predictions = []
    inf_times = []

    for image in images:
        filename, img_bytes = image
        inference_start_time = time.time()
        result = predict_fn(tf.convert_to_tensor([img_bytes], dtype=tf.string))
        end_time = time.time()

        predictions.append(
            {
                "filename": filename,
                "boxes": process_detection_result(result),
            }
        )
        inference_time = end_time - inference_start_time
        inf_times.append(inference_time)

    avg_inf_time = sum(inf_times) / len(inf_times)

    data = {
        "predictions": predictions,
        "inf_time": inf_times,
        "avg_inf_time": str(avg_inf_time),
    }

    return data


def get_current_timestamp():
    now = (
        datetime.datetime.utcnow()
    )  # using UTC time both on client and server for consistency
    timestamp_str = now.strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )  # for easy conversion to Python datetime https://stackoverflow.com/a/10805633/13727176
    return now, timestamp_str


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


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8502)
