#!/bin/bash
# starts a docker container with the Tensorflow Serving image and mounts a pretrained model
# assumes that save_pretrained_model.py has been run with the path to the pretrained model as argument
# TODO: parameterize this script, or figure out how to make all models in saved_models folder available
# TODO: convert to dockerfile

# Location of saved models
MODELS_PATH="$(pwd)/saved_models"
SSD_MODEL_PATH="$MODELS_PATH/ssd_mobilenet_v2"
RESNET_MODEL_PATH="$MODELS_PATH/resnet50_v1_fpn_640x640"

# Choose model to serve
MODEL_PATH=$RESNET_MODEL_PATH
MODEL_NAME="$(basename "$MODEL_PATH")"

docker run -it --rm -p 8500:8500 -p 8501:8501 -v "$MODEL_PATH:/models/$MODEL_NAME" -e MODEL_NAME=$MODEL_NAME tensorflow/serving:latest-gpu # use tensorflow/serving:latest if GPU not available