FROM tensorflow/tensorflow:latest-gpu

COPY ./requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY ./models/ /app/models/
COPY ./app.py /app/app.py

WORKDIR /app

RUN ls

# Expose port where server listens on
EXPOSE 8502

ENTRYPOINT ["python3"]
CMD ["app.py"]

