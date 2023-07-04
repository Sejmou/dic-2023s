# replace tensorflow/serving:latest-gpu with tensorflow/serving:latest if docker container has no GPU
FROM tensorflow/serving:latest-gpu

COPY ./config /config
COPY ./scripts/get_pretrained_models.py /scripts/get_pretrained_models.py
COPY ./scripts/requirements.txt /scripts/requirements.txt
COPY ./scripts/start_server.sh /scripts/start_server.sh

# Not sure if this would work on AWS (where should saved_models come from?)
# COPY ./saved_models /models

RUN chmod +x /scripts/get_pretrained_models.py

ENTRYPOINT ["/scripts/start_server.sh"]

EXPOSE 8500
EXPOSE 8501
