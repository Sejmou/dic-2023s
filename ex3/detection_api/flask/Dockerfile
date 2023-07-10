FROM tensorflow/tensorflow:latest-gpu-py3

COPY ./requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY ./models/ /app/models/
COPY ./app.py /app/app.py

RUN ls

WORKDIR /app

ENTRYPOINT ["python3"]
CMD ["app.py"]

