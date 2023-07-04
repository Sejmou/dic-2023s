# Saves pretrained object recognition models from TensorFlow Hub to a local directory, if they don't already exist.

# Those pretrained models can then be loaded/used by a TensorFlow Serving container.
# For every model, two versions are saved:
# - one version that accepts base64 encoded images as input
# - one version that accepts raw images as input
# %%
import tensorflow_hub as hub
import tensorflow as tf
import os
import pprint

# the URLs of the TF2 object detection models (from TensorFlow Hub) to download/save so that they can be deployed in the TF Serving container
# be careful to use TensorFlow v2 models! e.g. you cannot use this: https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1
module_handles = [
    "https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_640x640/1",  # larger model, slower but more accurate
    "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2",  # smaller model, faster but less accurate - should be 'fast enough' without GPU
]


def get_input_type_and_shape(signature):
    first_input = signature.inputs[0]
    try:
        input_type = first_input.dtype
    except:
        input_type = type(first_input)
    try:
        input_shape = first_input.shape  # shape prop might not exist
    except:
        input_shape = None

    return {"type": input_type, "shape": input_shape}


def get_model_path(model_name, version=1):
    path = os.path.join(
        os.getcwd(), "models", model_name, str(version)
    )  # version number is required for things to work out-of-the-box with tensorflow serving
    return path


def get_model_inputs(model):
    """
    IIUC, TF models can have multiple signatures, each with different input shapes (e.g. a object detection model could accept images both as base64 string AND tensor).

    This is a helper function to get the input type and shape for each signature of a model.

    The TF Hub models seem to only have a single 'serving_default' signature, but this function is still useful to get the input type and shape for that signature.
    """
    signature_input_dict = {}
    for name, signature in signatures.items():
        signature_input_dict[name] = get_input_type_and_shape(signature)
        print("Found following signatures and expected input shapes:")
        pp.pprint(signature_input_dict)


# taken from https://cloud.google.com/blog/topics/developers-practitioners/add-preprocessing-functions-tensorflow-models-and-deploy-vertex-ai?hl=en
def _preprocess(bytes_inputs):
    decoded = tf.io.decode_jpeg(bytes_inputs, channels=3)
    resized = tf.image.resize(decoded, size=(512, 512))
    return tf.cast(resized, dtype=tf.uint8)


def _get_serve_image_fn(model):
    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def serve_image_fn(bytes_inputs):
        decoded_images = tf.map_fn(_preprocess, bytes_inputs, dtype=tf.uint8)
        return model(decoded_images)

    return serve_image_fn


pp = pprint.PrettyPrinter(indent=4)

for module_handle in module_handles:
    model_name = module_handle.split("/")[-2]
    print(f"Processing model '{model_name}'")
    model_path = get_model_path(model_name)
    base64_model_path = get_model_path(model_name + "_base64")

    if not os.path.exists(base64_model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
            print(f"Downloading model from '{module_handle}'")
            detector = hub.load(module_handle)
            print(f"Saving model to '{model_path}'")
            tf.saved_model.save(detector, model_path)
        else:
            print(
                f"Base model already exists at '{model_path}', however version with base64 input still has to be created."
            )

        model = tf.saved_model.load(model_path)
        signatures = {
            "serving_default": _get_serve_image_fn(model).get_concrete_function(
                tf.TensorSpec(shape=[None], dtype=tf.string)
            )
        }
        print(f"Saving model with base64 image input to '{base64_model_path}'")
        tf.saved_model.save(model, base64_model_path, signatures=signatures)

    else:
        print(f"Model with base64 input already exists at '{base64_model_path}'")


# %%
