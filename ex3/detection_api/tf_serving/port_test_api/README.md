# Basic API
This is a basic dummy API to test whether an AWS EC2 instance is accessible via the web. It starts a Flask server on port 8501.

## Instructions
Launch EC2 instance, make sure you can SSH into it, copy this folder with `scp`, navigate into it.


### Option A: Basic Setup
Install Flask

Run app.py

### Option B: Docker Setup
## Install Docker
```bash
./install_docker.sh
```

## Build docker image from Dockerfile
```bash
docker build -t test .
```

##  Run docker container
```bash
# -p 8501:8501 exposes port 8501 to port 8501 on host
docker run -d -p 8501:8501 test
```