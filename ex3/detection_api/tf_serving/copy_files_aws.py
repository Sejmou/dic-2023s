import argparse
import os
import subprocess


def copy_files(pem_file, remote_address, source_path, destination_path):
    command = (
        f'scp -i "{pem_file}" -r {source_path} {remote_address}:{destination_path}'
    )
    subprocess.run(command, shell=True, check=True)


def main(pem_file, remote_address, cpu_only, local_models_path):
    config_source = "config"
    config_dest = "config"

    if cpu_only:
        script_source = "start_cpu.sh"
    else:
        script_source = "start_gpu.sh"

    script_dest = "start.sh"

    if os.path.isdir(local_models_path):
        copy_files(pem_file, remote_address, local_models_path, "models")
        print("Copied models directory to remote instance.")
    else:
        raise ValueError(
            "Invalid path to local models directory. Make sure to create it with `get_pretrained_models.py` first."
        )

    copy_files(pem_file, remote_address, config_source, config_dest)
    print("Copied config directory to remote instance.")
    copy_files(pem_file, remote_address, script_source, script_dest)
    print("Copied start script to remote instance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copies files required for running TensorFlow Serving Object Detection API to remote Ubuntu EC2 instance."
    )
    parser.add_argument(
        "-p",
        "--pem-file",
        default="aws-sejmou-ubuntu.pem",
        help="Path to the PEM file to use for SSH authentication",
    )
    parser.add_argument(
        "-r",
        "--remote-address",
        required=True,
        help="Public IPv4 DNS address of EC2 instance",
    )
    parser.add_argument(
        "-c",
        "--cpu-only",
        action="store_true",
        help="Create script for launching Docker container without GPU support",
    )
    parser.add_argument(
        "-l",
        "--local-models-path",
        help="Path to local models directory",
        required=True,
    )

    args = parser.parse_args()

    main(
        args.pem_file,
        "ubuntu@" + args.remote_address,
        args.cpu_only,
        args.local_models_path,
    )
