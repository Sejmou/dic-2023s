# Flask Object Detection API

## Installation steps

### Install nvidia-containertoolkit on host machine
Find instructions on Google (can't remember what I did lol)

### Build docker image from Dockerfile
For some reason, the following commands only worked w/ sudo on my local Ubuntu Machine

```bash
sudo docker build -t flask-detection-api .
```

### Run docker container locally
Pass `nvidia` as runtime argument to enable GPU, expose port 8502 from container to host machine

```bash
sudo docker run -it --runtime=nvidia -p 8502:8502 flask-detection-api
```

