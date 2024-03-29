# Download the TensorFlow Serving Docker image and repo
docker pull tensorflow/serving:latest-gpu

# Set up the environment variables for model and config folders
MODELS_FOLDER=$(pwd)/models
CONFIG_FOLDER=$(pwd)/config

# Start TensorFlow Serving container with the following additional configuration:
# use all available GPUs
# expose the ports for gRPC (8500) REST API (8501) to the host machine with the -p flag
# mount the folders with the models and the config to the container
# set the path to the model config file (will be used to configure the API endpoints for each model)
docker run -t --rm --gpus all \
    -p 8501:8501 -p 8500:8500 \
    -v "$MODELS_FOLDER:/models/" -v "$CONFIG_FOLDER/:/config/" \
    tensorflow/serving:latest-gpu \
    --model_config_file=/config/models.config