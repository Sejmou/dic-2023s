# Object Detection API built with TensorFlow Serving (using pretrained models from TensorFlow Hub)
This API utilizes Object Detection Models hosted on TensorFlow Hub.

In fact, it should work with _any_ object detection model saved as a TensorFlow v2 SavedModel. However, for now it is configured to load two models (a fast, but less accurate one and a slow, more accurate one) from TensorFlow Hub and provide them on different endpoints. The models were adapted slightly so that they accept Base64 encoded JPEG images (instead of a tensor of shape) as input.

See also the `get_pretrained_models.py` script and `models.config`.

More README might follow some day lol


## Local Setup
```bash
# create Docker image called object-detection-api
docker build -f Dockerfile -t object-detection-api .
```

```bash
# run Docker container (IMPORTANT: use -p to publish ports to host as well (otherwise they are only exposed within the container network!)
docker run -t object-detection-api -p 8500:8500 8501:8501
```

## AWS Setup
There's probably lots of ways to get this done. The approach explained here runs the Object Detection API using a Docker Image hosted on an Elastic Cloud Compute (EC2) instance, pulling the Docker Image from Elastic Container Registry (ECR, the Amazon equivalent of DockerHub).

### Get to AWS Management Console (with AWS Learner Lab Account)

1. Login to Learner Lab
2. Navigate to Dashboard
3. Click 'Start'
4. When 'light'/indicator next to link called 'AWS' becomes green, click it to get to AWS console

### Uploading Docker Image to ECR
#### Create container repository
In AWS Management console:
1. Search for ECR in searchbar
2. Click link
3. Navigate to 'Repositories'
4. Click on 'Create repository'
5. Create private or public repository called 'object-detection-api'

#### Push Docker Image
First, install and configure [aws-cli]([https://](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)) on your local device.

Then, follow the steps listed in 'View push commands' for your created ECS repository in AWS Management console to create and upload the Docker image for this project to the repository. Run them on your local machine's terminal from this directory. Note: The docker image creation and especially pushing takes quite some time.

### Launching EC2 instance with the Docker Image
The g4dn instances are probably most suitable for this application as they also come with a GPU. Pick an Ubuntu AMI (probably exact choice doesn't matter as Docker is used anyway - however, there's also a TensorFlow Deep Learning AMI). The smallest/cheapest one is `g4dn.large` (hourly cost of less than a dollar per hour).

> **Important Note**: the g4dn.large image does not work with the Learner Lab AWS account. There, only a small subset of all available EC2 instances can be used (due to restrictions on the default service quotas). Even with a private account, you will need to request service quota increases (specifically, something w/ 'number of vCPUs for G, ... type instances' needs to be set to at least 4 as g4dn.large has 4 vCPUs; maybe other increases are also required). This process can take one or two work days, unfortunately.

## Additional AWS instructions

### Access to AWS CLI from local device with Learner Lab account
In Learner Lab Dashboard:
1. Click on 'AWS Details'
2. Click on 'Show' button next to 'AWS CLI'
3. Follow instructions (copying the shown config to the specified folder)