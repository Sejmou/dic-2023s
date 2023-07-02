# Saves a pretrained model from TensorFlow Hub to a local directory
# This pretrained model can then be loaded/used by a TensorFlow Serving container

import tensorflow_hub as hub
import tensorflow as tf
import os

# the URL to the TF2 object detection model (from TensorFlow Hub) to use
# be careful to use TensorFlow v2 models! e.g. you cannot use this: https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1
# module_handle = "https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_640x640/1"  # larger model, slower but more accurate
module_handle = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"  # smaller model, faster but less accurate - should be 'fast enough' without GPU
model_name = module_handle.split("/")[-2]

model_path = os.path.join(
    os.getcwd(), "saved_models", model_name, "1"
)  # the 1 is the version number (required by TF Serving)

if not os.path.exists(model_path):
    os.makedirs(model_path, exist_ok=True)
    detector = hub.load(module_handle)
    print(f"Saving model to '{model_path}'")
    tf.saved_model.save(detector, model_path)

else:
    print(f"Model already exists at '{model_path}'")
