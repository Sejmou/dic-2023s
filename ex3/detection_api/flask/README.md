# Flask Object Detection API

## Local Setup

### Install nvidia-containertoolkit on host machine
Find instructions on Google (can't remember what I did lol)

### Build docker image from Dockerfile
For some reason, the following commands only worked w/ sudo on my local Ubuntu Machine

```bash
sudo docker build -t flask-detection-api .
```

### Run Docker Container
Pass `nvidia` as runtime argument to enable GPU, expose port 8502 from container to host machine

```bash
sudo docker run -it --runtime=nvidia -p 8502:8502 flask-detection-api
```

## AWS Setup

### Upload local Docker Image to ECR
ECR is similar to DockerHub, i.e. you can upload and version Docker Images in repositories. They can then be used by AWS EC2 instances or also with AWS ECS (which is a container cluster auto-scaling solution provided by AWS).

#### Create container repository
In AWS Management console:
1. Search for ECR in searchbar
2. Click link
3. Navigate to 'Repositories'
4. Click on 'Create repository'
5. Create private or public repository called 'object-detection-api'

#### Push Docker Image
First, install and configure [aws-cli]([https://](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)) on your local device.

Then, follow the steps listed in 'View push commands' for your created ECS repository in AWS Management console to create and upload the Docker image for this project to the repository. Run them on your local machine's terminal from this directory. 
> **Note:** The docker image creation and especially pushing takes quite some time. Also, for some reason, the docker commands didn't work for me without `sudo`. I suppose this has something to do with the fact that I have installed both regular docker and docker stuff that comes with Docker Desktop for Linux.

Now, the image can be pulled from anywhere by using `docker pull` with the path to the repository.

### Pick EC2 instance type
The [g4dn](https://aws.amazon.com/ec2/instance-types/g4/#:~:text=G4dn%20instances%2C%20powered,such%20as%20CUDA) instances are probably most suitable for this application as they also come with a GPU. Preferrably, use the [Deep Learning AMI GPU TensorFlow 2.12.0 AMI with Ubuntu 20.04](https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-tensorflow-2-12-ubuntu-20-04/) as it includes Docker and it was also setup properly for CUDA. The smallest/cheapest one is `g4dn.large` (hourly cost of less than a dollar per hour).

> **Important Note**: the g4dn.large image does not work with the Learner Lab AWS account. There, only a small subset of all available EC2 instances can be used (due to restrictions on the default service quotas). Even with a private account, you will need to request service quota increases (specifically, something w/ 'number of vCPUs for G, ... type instances' needs to be set to at least 4 as g4dn.large has 4 vCPUs; maybe other increases are also required). This process can take one or two work days, unfortunately.

### Create EC2 instance
Follow instructions from AWS Managment Console. Make sure to also specify (optionally create and download) a key pair so that you can easily ssh into the EC2 instance later. 

### Configure security group (inbound and outbound rules)
Make sure to also make the instance reachable via HTTP(S) by configuring and selecting the correct security group for the EC2 instance and configuring it accordingly. The security group settings can be found in the VPC (Virtual Private Cloud) section of the AWS Managment Console. Settings that should work:

- inbound: 
  - TCP from anywhere via IPv4 and IPv6 on port 8502
  - SSH from anywhere via IPv4 and IPv6 on port 22
- outbound: you can keep the default (all traffic)

You can also create a launch template from the created container so that you don't have to reconfigure things every time you launch a new instance.

### Setup EC2 Instance

#### Login via SSH
Check the management console (instance summary -> 'Connect') for the command. Should be something like:
```bash
ssh -i "your-keyfile.pem" ubuntu@ec2-3-122-229-105.eu-central-1.compute.amazonaws.com
```
replacing keyfile and adress with your own path, obviously. If this works, you have successfully connected with the instance.

This will take a bit as the Docker Image needs to be pulled. But after a while you should see a message that the API is running on port 8501. The API should then be accessible.

#### Pull image from ECR
We have already pushed a version of the Docker Image to a public ECR repository. This command will pull it to the EC2 instance.

```bash
docker pull public.ecr.aws/k6o2m4v2/flask-detection-api:latest
```

#### Run Docker Image
The command is almost the same as with the local setup:

```bash
docker run -it --runtime=nvidia --expose 8502 -p 8502:8502 public.ecr.aws/k6o2m4v2/flask-detection-api:latest
```

Once you see a message that the server is running, the EC2 instance should be accessible via the public IP of the EC2 instance under port 8502 and the `client_flask_api.py` script should work.