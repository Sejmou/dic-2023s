# Object Detection API built with TensorFlow Serving (using pretrained models from TensorFlow Hub)
This API utilizes Object Detection Models hosted on TensorFlow Hub.

In fact, it should work with _any_ object detection model saved as a TensorFlow v2 SavedModel. However, for now it is configured to load two models (a fast, but less accurate one and a slow, more accurate one) from TensorFlow Hub and provide them on different endpoints. The models were adapted slightly so that they accept Base64 encoded JPEG images (instead of a tensor of shape) as input.

See also the `get_pretrained_models.py` script and `models.config`.

More README might follow some day lol

## How to run
```
# create Docker image called object-detection-api
docker build -f Dockerfile -t object-detection-api .
```

```
# run Docker container (IMPORTANT: use -p to publish ports to host as well (otherwise they are only exposed within the container network!)
docker run -t object-detection-api -p 8500:8500 8501:8501
```