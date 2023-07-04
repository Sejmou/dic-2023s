# replace tensorflow/serving:latest-gpu with tensorflow/serving:latest if docker container has no GPU
FROM tensorflow/serving:latest-gpu

COPY ./config /config
COPY ./scripts/get_pretrained_models.py /scripts/get_pretrained_models.py
COPY ./scripts/requirements.txt /scripts/requirements.txt
COPY ./scripts/start_server.sh /scripts/start_server.sh

# Not sure if this would work on AWS (where should saved_models come from?)
# COPY ./saved_models /models

RUN chmod +x /scripts/get_pretrained_models.py

# For some reason, I couldn't run the pip install with requirements.txt and  Python script for downloading the pretrained models in a RUN command
ENTRYPOINT ["/scripts/start_server.sh"]

# IMPORTANT: using expose here does NOT publish the ports to the host machine (so, you would not be able to access the server from outside the container)
# "To actually publish the port when running the container, use the -p flag on docker run to publish and map one or more ports" https://docs.docker.com/engine/reference/builder/#expose
# Could also use the -P flag to publish all exposed ports. However, they are then published to *random* ports on the host machine
EXPOSE 8500
EXPOSE 8501