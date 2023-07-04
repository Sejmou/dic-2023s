#!/bin/bash
set -xe \
 && apt-get update \
 && apt-get install python3-pip -y \
 && pip install -r /scripts/requirements.txt \
 && python3 /scripts/get_pretrained_models.py

# Start the tensorflow serving server
# need to expose REST API explicitly as we are not just using docker run: https://www.tensorflow.org/tfx/serving/api_rest#start_modelserver_with_the_rest_api_endpoint
# for some reason, tensorflow_model_server --model_config_file=/config/models.config --rest_api_port 8501 did not work (unknown argument --rest_api_port)
tensorflow_model_server --rest_api_port=8501 \
  --model_config_file=/config/models.config
#  --model_config_file_poll_wait_seconds=60 could use this if models are being updated on the fly (not the case here)