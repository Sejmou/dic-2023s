# Object Detection API built with TensorFlow Serving (using pretrained models from TensorFlow Hub)
This API utilizes Object Detection Models hosted on TensorFlow Hub.

In fact, it should work with _any_ object detection model saved as a TensorFlow v2 SavedModel. However, for now it is configured to load two models (a fast, but less accurate one and a slow, more accurate one) from TensorFlow Hub and provide them on different endpoints. The models were adapted slightly so that they accept Base64 encoded JPEG images (instead of a tensor of shape) as input.

See also the `get_pretrained_models.py` script and `models.config`.

More README might follow some day lol

## How to run

### Local setup
```
# create Docker image called object-detection-api
docker build -f Dockerfile -t object-detection-api .
```

```
# run Docker container (IMPORTANT: use -p to publish ports to host as well (otherwise they are only exposed within the container network!)
docker run -t object-detection-api -p 8500:8500 8501:8501
```

### AWS (with Learner Lab Account)
There's probably lots of ways to get this done

#### Pt. 1: Get to AWS Console (web browser)
1. Login to Learner Lab
2. Navigate to Dashboard
3. Click 'Start'
4. When 'light'/indicator next to link called 'AWS' becomes green, click it to get to AWS console

#### Pt. 2: Setup container registry in ECR
1. Search for ECR in searchbar
2. Click link
3. Navigate to 'Repositories'
4. Click on 'Create repository'
5. Create private repository called 'object-detection-api'

#### Pt. 3 (not sure if actually required): Create and push Docker Image to ECR
First, install and configure [aws-cli]([https://](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)) on your local device.

Then, find credentials in Learner Lab Dashboard:
1. Click on 'AWS Details'
2. Click on 'Show' button next to 'AWS CLI'
3. Follow instructions (copying the shown config to the specified folder)

You should now be able to use the AWS CLI with the account provided by the learner lab from your local machine.

Finally, build and push Docker Image to ECR by following steps listed in 'View push commands' for your created ECS repository in AWS console (run them on your local machine's terminal from this directory). Note: The docker image creation and especially pushing takes quite some time.

#### Pt. 4: Launch g4dn.xlarge EC2 instance with the Docker Image
TODO: detail steps for creating/launching EC2 instance
Note: the g4dn.large image does not work with the Learner Lab AWS account. There, only basic CPU only instances can be used (I have no idea which ones work and which don't - in any case, you would probably need to adapt the Docker Image to use tensorflow-serving:latest instead of tensorflow-serving:gpu-latest)

TODO: figure out how to either pull Docker Image uploaded in Pt. 3 OR copy dockerfile to EC2 instance and build Docker Image there