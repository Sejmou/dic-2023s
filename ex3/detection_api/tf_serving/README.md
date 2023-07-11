# Object Detection API built with TensorFlow Serving (using pretrained models from TensorFlow Hub)
This API utilizes Object Detection Models hosted on TensorFlow Hub and serves them via a TensorFlow Serving Docker Image.

In fact, it should work with _any_ object detection model saved as a TensorFlow v2 SavedModel. However, for now it is configured to use two models (a fast, but less accurate one and a slow, more accurate one) from TensorFlow Hub, serving them on different endpoints. 

## How it works
The `get_pretrained_models.py` script in the `model_fetching` directory downloads the models, adapts them so that they accept Base64 encoded JPEG images (instead of a tensor of shape) as input, and finally stores them.

A TensorFlow Serving Docker Image is used to serve the models on the host machine, making them accessible via a REST API. The `models.config` is used to map the folders with the saved models to API endpoints (details on how that works [here](https://www.tensorflow.org/tfx/serving/serving_config#model_server_config_details)).

The `run_cpu.sh` and `run_gpu.sh` scripts in this folder can be used to set up and start the server Docker Image, either with CPU only or with GPU (depending on the host machine). The host machine is expected to come with Python 3.8+ preinstalled and use 64-bit x86 chip architecture, which means that this will not run out-of-the-box on ARM architectures, including M1/M2 Macs. Note that the scripts expect that a `models` directory created with the `get_pretrained_models.py` script exists (it should contain the models referred to in `models.config`).

## Local Setup
Just run either `run_cpu.sh` or `run_gpu.sh` (`chmod +x` might be necessary).

## AWS Setup
There's probably lots of ways to get this done. The approach explained here runs the same code as in the local setup on an Elastic Cloud Compute (EC2) instance with some additional config to make the object detection API publicly available from anywhere on the web via its IP.

### Pick EC2 instance type
The g4dn instances are probably most suitable for this application as they also come with a GPU. Pick an Ubuntu AMI (probably exact choice doesn't matter as Docker is used anyway - however, there's also a TensorFlow Deep Learning AMI). The smallest/cheapest one is `g4dn.large` (hourly cost of less than a dollar per hour).

> **Important Note**: the g4dn.large image does not work with the Learner Lab AWS account. There, only a small subset of all available EC2 instances can be used (due to restrictions on the default service quotas). Even with a private account, you will need to request service quota increases (specifically, something w/ 'number of vCPUs for G, ... type instances' needs to be set to at least 4 as g4dn.large has 4 vCPUs; maybe other increases are also required). This process can take one or two work days, unfortunately.

### Create EC2 instance
Follow instructions from AWS Managment Console. Make sure to also specify (optionally create and download) a key pair so that you can easily ssh into the EC2 instance later. 

Make sure to also make the instance reachable via HTTP(S) by configuring and selecting the correct security group. Settings that should work in theory:
- inbound: 
  - TCP from anywhere via IPv4 and IPv6 on port 8501
  - SSH from anywhere via IPv4 and IPv6 on port 22
- outbound: doesn't really matter for now I guess (need further research to figure out meaningful values)

**NOTE:** In practice, it only worked for me once I allowed _any_ TCP traffic (on all ports) - absolutely no idea why 🤷🏼‍♂️

You can also create a launch template from the created container so that you don't have to reconfigure things every time you launch a new instance.

### Setup EC2 Instance via SSH
Check the management console (instance summary -> 'Connect') for the command. Should be something like:
```bash
ssh -i "~/.ssh/aws-sejmou.pem" ubuntu@ec2-3-122-229-105.eu-central-1.compute.amazonaws.com
```
replacing keyfile and adress with your own path, obviously.

If this works, you have successfully connected with the instance and you should be able to run the `copy_files_aws.py` script on your local machine which copies over all the necessary files via `scp`:
```bash
python copy_files_aws.py -r ec2-3-70-137-214.eu-central-1.compute.amazonaws.com
```
Check the script for details on what happens and what arguments are required/expected.

Once the script is done, all the necessary things to run the API server are there! To start the server, run the server startup script from the instance you ssh-ed into:
```bash
./start.sh
```

This will take a bit as the Docker Image needs to be pulled. But after a while you should see a message that the API is running on port 8501 of localhost. The API should then be accessible.

### Test Connectivity
```bash
curl ec2-3-122-229-105.eu-central-1.compute.amazonaws.com/v1/models/resnet50_v1_fpn_640x640/metadata
```

## Additional AWS instructions

### Get to AWS Management Console (with AWS Learner Lab Account)

1. Login to Learner Lab
2. Navigate to Dashboard
3. Click 'Start'
4. When 'light'/indicator next to link called 'AWS' becomes green, click it to get to AWS console
### Access to AWS CLI from local device with Learner Lab account
In Learner Lab Dashboard:
1. Click on 'AWS Details'
2. Click on 'Show' button next to 'AWS CLI'
3. Follow instructions (copying the shown config to the specified folder)

### Store credentials for root user of private AWS account for use with AWS CLI
1. Follow instructions for creating access key [here](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_root-user.html#id_root-user_manage_add-key)
2. Use `aws config` to store the created access key (+ id) in the credentials file (automatically used by CLI). Recommended settings:
   - default region name: eu-central-1 (Frankfurt, closest to Vienna)
   - default output format: json