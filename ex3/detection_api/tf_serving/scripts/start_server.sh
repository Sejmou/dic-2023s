#!/bin/bash
set -xe \
 && apt-get update \
 && apt-get install python3-pip -y \
 && pip install -r /scripts/requirements.txt \
 && python3 /scripts/get_pretrained_models.py

tensorflow_model_server --model_config_file=/config/models.config
#  --model_config_file_poll_wait_seconds=60 could use this if models are being updated on the fly (not the case here)